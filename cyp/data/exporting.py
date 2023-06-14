import ee
import ssl
import time
from pathlib import Path
import numpy as np

from .utils import load_clean_yield_data as load
from .utils import get_tif_files
from .. import MAJOR_STATES


class MODISExporter:
    """A class to export MODIS data from
    the Google Earth Engine to Google Drive

    Parameters
    ----------

    locations_filepath: pathlib Path, default=Path('data/yield_data.csv')
        A path to the yield data
    collection_id: str, default='MODIS/051/MCD12Q1'
        The ID Earth Engine Image Collection being exported
    """

    # CS4245: load locations differently in case the Italian data is used
    def __init__(
        self,
        locations_filepath=Path("data/yield_data.csv"),
        collection_id="MODIS/051/MCD12Q1",
        for_italy=False
    ):
        self.locations = load(locations_filepath, for_italy=for_italy)

        self.collection_id = collection_id

        try:
            ee.Initialize()
            print("The Earth Engine package initialized successfully!")
        except ee.EEException:
            print(
                "The Earth Engine package failed to initialize! "
                "Have you authenticated the earth engine?"
            )

    # CS4245: load locations differently in case the Italian data is used
    def update_parameters(self, locations_filepath=None, collection_id=None, for_italy=False):
        """
        Update the locations file or the collection id
        """
        if locations_filepath is not None:
            self.locations = load(locations_filepath, for_italy=for_italy)
        if collection_id is not None:
            self.collection_id = collection_id

    @staticmethod
    def _export_one_image(img, folder, name, region, scale, crs):
        # export one image from Earth Engine to Google Drive
        # Author: Jiaxuan You, https://github.com/JiaxuanYou
        print(f"Exporting to {folder}/{name}")
        task_dict = {
            "driveFolder": folder,
            "driveFileNamePrefix": name,
            "scale": scale,
            "crs": crs,
        }
        if region is not None:
            task_dict.update({"region": region})
        task = ee.batch.Export.image(img, name, task_dict)
        task.start()
        while task.status()["state"] == "RUNNING":
            print("Running...")
            # Perhaps task.cancel() at some point.
            time.sleep(10)

        print(f"Done: {task.status()}")

    def export_us(
        self,
        folder_name,
        data_type,
        coordinate_system="EPSG:4326",
        scale=500,
        export_limit=None,
        min_img_val=None,
        max_img_val=None,
        major_states_only=True,
        check_if_done=False,
        download_folder=None,
    ):
        """Export an Image Collection from Earth Engine to Google Drive

        Parameters
        ----------
            folder_name: str
                The name of the folder to export the images to in
                Google Drive. If the folder is not there, this process
                creates it
            data_type: str {'image', 'mask', 'temperature'}
                The type of data we are collecting. This tells us which bands to collect.
            coordinate_system: str, default='EPSG:4326'
                The coordinate system in which to export the data
            scale: int, default=500
                The pixel resolution, as determined by the output.
                https://developers.google.com/earth-engine/scale
            export_limit: int or None, default=None
                If not none, limits the number of files exported to the value
                passed.
            min_img_val = int or None:
                A minimum value to clip the band values to
            max_img_val: int or None
                A maximum value to clip the band values to
            major_states_only: boolean, default=True
                Whether to only use the 11 states responsible for 75 % of national soybean
                production, as is done in the paper
            check_if_done: boolean, default=False
                If true, will check download_folder for any .tif files which have already been
                downloaded, and won't export them again. This effectively allows for
                checkpointing, and prevents all files from having to be downloaded at once.
            download_folder: None or pathlib Path, default=None
                Which folder to check for downloaded files, if check_if_done=True. If None, looks
                in data/folder_name
        """
        if check_if_done:
            if download_folder is None:
                download_folder = Path("data") / folder_name
                already_downloaded = get_tif_files(download_folder)

        imgcoll = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(ee.Geometry.Rectangle(-106.5, 50, -64, 23))
            .filterDate("2002-12-31", "2016-8-4")
        )

        datatype_to_func = {
            "image": _append_im_band,
            "mask": _append_mask_band,
            "temperature": _append_temp_band,
        }

        img = imgcoll.iterate(datatype_to_func[data_type])
        img = ee.Image(img)

        # "clip" the values of the bands
        if min_img_val is not None:
            # passing en ee.Number creates a constant image
            img_min = ee.Image(ee.Number(min_img_val))
            img = img.min(img_min)
        if max_img_val is not None:
            img_max = ee.Image(ee.Number(max_img_val))
            img = img.max(img_max)

        # note that the county regions are pulled from Google's Fusion tables. This calls a merge
        # of county geometry and census data:
        # https://fusiontables.google.com/data?docid=1S4EB6319wWW2sWQDPhDvmSBIVrD3iEmCLYB7nMM#rows:id=1

        region = ee.FeatureCollection("TIGER/2018/Counties")

        # turn the strings into numbers, see
        # https://developers.google.com/earth-engine/datasets/catalog/TIGER_2018_Counties
        def county_to_int(feature):
            return feature.set("COUNTYFP", ee.Number.parse(feature.get("COUNTYFP")))

        def state_to_int(feature):
            return feature.set("STATEFP", ee.Number.parse(feature.get("STATEFP")))

        region = region.map(county_to_int)
        region = region.map(state_to_int)

        count = 0

        for state_id, county_id in np.unique(
            self.locations[["State ANSI", "County ANSI"]].values, axis=0
        ):
            if major_states_only:
                if int(state_id) not in MAJOR_STATES:
                    print(f"Skipping state id {int(state_id)}")
                    continue

            fname = "{}_{}".format(int(state_id), int(county_id))

            if check_if_done:
                if f"{fname}.tif" in already_downloaded:
                    print(f"{fname}.tif already downloaded! Skipping")
                    continue

            file_region = region.filterMetadata(
                "COUNTYFP", "equals", int(county_id)
            ).filterMetadata("STATEFP", "equals", int(state_id))
            file_region = ee.Feature(file_region.first())
            processed_img = img.clip(file_region)
            file_region = None
            while True:
                try:
                    self._export_one_image(
                        processed_img,
                        folder_name,
                        fname,
                        file_region,
                        scale,
                        coordinate_system,
                    )
                except (ee.ee_exception.EEException, ssl.SSLEOFError):
                    print(f"Retrying State {int(state_id)}, County {int(county_id)}")
                    time.sleep(10)
                    continue
                break

            count += 1
            if export_limit:
                if count >= export_limit:
                    print("Reached export limit! Stopping")
                    break
        print(f"Finished Exporting {count} files!")

    ### [CS4245] ###
    #
    #   Separate export function for Italian sattellite data
    #
    ################
    def export_italy(
        self,
        folder_name,
        data_type,
        coordinate_system="EPSG:4326",
        scale=500,
        export_limit=None,
        min_img_val=None,
        max_img_val=None
    ):
        imgcoll = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(ee.Geometry.Rectangle(4.715739, 47.523461, 20.653583, 35.939656))
            .filterDate("2002-12-31", "2016-8-4")
        )

        datatype_to_func = {
            "image": _append_im_band,
            "mask": _append_mask_band,
            "temperature": _append_temp_band,
        }

        img = imgcoll.iterate(datatype_to_func[data_type])
        img = ee.Image(img)

        # "clip" the values of the bands
        if min_img_val is not None:
            # passing en ee.Number creates a constant image
            img_min = ee.Image(ee.Number(min_img_val))
            img = img.min(img_min)
        if max_img_val is not None:
            img_max = ee.Image(ee.Number(max_img_val))
            img = img.max(img_max)

        country = ee.FeatureCollection("FAO/GAUL/2015/level2").filter("ADM0_NAME == 'Italy'")
        count = 0

        # Italy province codes
        adm_codes = [ [ 1631, 18400 ], [ 1631, 18407 ], [ 1618, 18323 ], [ 1618, 18325 ], [ 1618, 18327 ], [ 1620, 18336 ], [ 1620, 18342 ], [ 1616, 18317 ], [ 1616, 18318 ], [ 1616, 18319 ], [ 1616, 18320 ], [ 1617, 18321 ], [ 1617, 18322 ], [ 1618, 18324 ], [ 1618, 18326 ], [ 1619, 18328 ], [ 1619, 18329 ], [ 1619, 18330 ], [ 1619, 18332 ], [ 1619, 18333 ], [ 1620, 18334 ], [ 1620, 18337 ], [ 1620, 18338 ], [ 1620, 18339 ], [ 1620, 18340 ], [ 1620, 18341 ], [ 1622, 18347 ], [ 1622, 18348 ], [ 1622, 18349 ], [ 1622, 18350 ], [ 1622, 18351 ], [ 1623, 18352 ], [ 1623, 18353 ], [ 1623, 18354 ], [ 1623, 18355 ], [ 1625, 18367 ], [ 1625, 18368 ], [ 1625, 18369 ], [ 1625, 18370 ], [ 1626, 18371 ], [ 1626, 18372 ], [ 1627, 18373 ], [ 1627, 18374 ], [ 1627, 18376 ], [ 1627, 18378 ], [ 1628, 18381 ], [ 1628, 18382 ], [ 1628, 18383 ], [ 1628, 18384 ], [ 1628, 18385 ], [ 1629, 18386 ], [ 1629, 18387 ], [ 1629, 18388 ], [ 1629, 18389 ], [ 1630, 18390 ], [ 1630, 18391 ], [ 1630, 18392 ], [ 1630, 18393 ], [ 1630, 18394 ], [ 1630, 18395 ], [ 1630, 18396 ], [ 1630, 18397 ], [ 1630, 18398 ], [ 1631, 18399 ], [ 1631, 18401 ], [ 1631, 18402 ], [ 1631, 18403 ], [ 1631, 18404 ], [ 1631, 18405 ], [ 1631, 18406 ], [ 1631, 18408 ], [ 1633, 18411 ], [ 1633, 18412 ], [ 1620, 18335 ], [ 1621, 18343 ], [ 1621, 18344 ], [ 1621, 18345 ], [ 1621, 18346 ], [ 1624, 18356 ], [ 1624, 18357 ], [ 1624, 18358 ], [ 1624, 18359 ], [ 1624, 18360 ], [ 1624, 18361 ], [ 1624, 18362 ], [ 1624, 18363 ], [ 1624, 18364 ], [ 1624, 18365 ], [ 1624, 18366 ], [ 1627, 18375 ], [ 1627, 18377 ], [ 1627, 18379 ], [ 1627, 18380 ], [ 1632, 18409 ], [ 1632, 18410 ], [ 1634, 18413 ], [ 1635, 18414 ], [ 1635, 18415 ], [ 1635, 18416 ], [ 1635, 18417 ], [ 1635, 18418 ], [ 1635, 18419 ], [ 1635, 18420 ] ]

        for [adm1_code, adm2_code] in adm_codes:
            fname = "{}_{}_{}".format(122, adm1_code, adm2_code)

            file_region = country.filter(f"ADM1_CODE == {adm1_code}").filter(f"ADM2_CODE == {adm2_code}")
            processed_img = img.clip(file_region)
            while True:
                try:
                    self._export_one_image(
                        processed_img,
                        folder_name,
                        fname,
                        file_region.geometry(),
                        scale,
                        coordinate_system,
                    )
                except (ee.ee_exception.EEException, ssl.SSLEOFError) as err:
                    print(f"Retrying (ADM1 {adm1_code} - ADM2 {adm2_code}), Italy")
                    time.sleep(10)
                    continue
                break

            count += 1
            if export_limit:
                if count >= export_limit:
                    print("Reached export limit! Stopping")
                    break
                
        print(f"Finished Exporting {count} files!")

    def export(
        self,
        folder_name,
        data_type,
        coordinate_system="EPSG:4326",
        scale=500,
        export_limit=None,
        min_img_val=None,
        max_img_val=None,
        major_states_only=True,
        check_if_done=False,
        download_folder=None,
        for_italy=False
    ):
        if for_italy: 
            self.export_italy(folder_name, data_type, coordinate_system, scale, export_limit, min_img_val, max_img_val)
        else: 
            self.export_us(folder_name, data_type, coordinate_system, scale, export_limit, min_img_val, max_img_val, major_states_only, check_if_done, download_folder)

    ### [CS4245] ###
    #
    #   Added for_italy parameter to indicate whether to export for Italy or US
    #
    ################
    def export_all(
        self,
        export_limit=None,
        major_states_only=True,
        check_if_done=True,
        download_folder=None,
        for_italy=False
    ):
        """
        Export all the data.

        download_folder = list of 3 pathlib Paths, for each of the 3 downloads
        """
        if download_folder is None:
            download_folder = [None] * 3
        assert (
            len(download_folder) == 3
        ), "Must have 3 download folders for the 3 exports!"

        # first, make sure the class was initialized correctly
        self.update_parameters(
            locations_filepath=Path("data/yield_data.csv"),
            collection_id="MODIS/MOD09A1",
            for_italy=for_italy
        )

        # # pull_MODIS_entire_county_clip.py
        self.export(
            folder_name="crop_yield-data_image",
            data_type="image",
            min_img_val=16000,
            max_img_val=100,
            export_limit=export_limit,
            major_states_only=major_states_only,
            check_if_done=check_if_done,
            download_folder=download_folder[0],
            for_italy=for_italy
        )

        # pull_MODIS_landcover_entire_county_clip.py
        self.update_parameters(collection_id="MODIS/006/MCD12Q1", for_italy=for_italy)
        self.export(
            folder_name="crop_yield-data_mask",
            data_type="mask",
            export_limit=export_limit,
            major_states_only=major_states_only,
            check_if_done=check_if_done,
            download_folder=download_folder[1],
            for_italy=for_italy
        )

        # pull_MODIS_temperature_entire_county_clip.py
        self.update_parameters(collection_id="MODIS/MYD11A2", for_italy=for_italy)
        self.export(
            folder_name="crop_yield-data_temperature",
            data_type="temperature",
            export_limit=export_limit,
            major_states_only=major_states_only,
            check_if_done=check_if_done,
            download_folder=download_folder[2],
            for_italy=for_italy
        )
        print("Done exporting! Download the folders from your Google Drive")


def _append_mask_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )


def _append_temp_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0, 4])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )


def _append_im_band(current, previous):
    # Transforms an Image Collection with 1 band per Image into a single Image with items as bands
    # Author: Jamie Vleeshouwer

    # Rename the band
    previous = ee.Image(previous)
    current = current.select([0, 1, 2, 3, 4, 5, 6])
    # Append it to the result (Note: only return current item on first element/iteration)
    return ee.Algorithms.If(
        ee.Algorithms.IsEqual(previous, None),
        current,
        previous.addBands(ee.Image(current)),
    )
