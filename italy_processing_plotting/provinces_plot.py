import torch
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path

mapping = [ [ 18400, "Firenze" ], [ 18407, "Prato" ], [ 18323, "Catanzaro" ], [ 18325, "Crotone" ], [ 18327, "Vibo Valentia" ], [ 18336, "Forli'" ], [ 18342, "Rimini" ], [ 18317, "Chieti" ], [ 18318, "L'Aquila" ], [ 18319, "Pescara" ], [ 18320, "Teramo" ], [ 18321, "Matera" ], [ 18322, "Potenza" ], [ 18324, "Cosenza" ], [ 18326, "Reggio Calabria" ], [ 18328, "Avellino" ], [ 18329, "Benevento" ], [ 18330, "Caserta" ], [ 18332, "Napoli" ], [ 18333, "Salerno" ], [ 18334, "Bologna" ], [ 18337, "Modena" ], [ 18338, "Parma" ], [ 18339, "Piacenza" ], [ 18340, "Ravenna" ], [ 18341, "Reggio Emilia" ], [ 18347, "Frosinone" ], [ 18348, "Latina" ], [ 18349, "Rieti" ], [ 18350, "Roma" ], [ 18351, "Viterbo" ], [ 18352, "Genova" ], [ 18353, "Imperia" ], [ 18354, "La Spezia" ], [ 18355, "Savona" ], [ 18367, "Ancona" ], [ 18368, "Ascoli Piceno" ], [ 18369, "Macerata" ], [ 18370, "Pesaro" ], [ 18371, "Campobasso" ], [ 18372, "Isernia" ], [ 18373, "Alessandria" ], [ 18374, "Asti" ], [ 18376, "Cuneo" ], [ 18378, "Torino" ], [ 18381, "Bari" ], [ 18382, "Brindisi" ], [ 18383, "Foggia" ], [ 18384, "Lecce" ], [ 18385, "Taranto" ], [ 18386, "Cagliari" ], [ 18387, "Nuoro" ], [ 18388, "Oristano" ], [ 18389, "Sassari" ], [ 18390, "Agrigento" ], [ 18391, "Caltanissetta" ], [ 18392, "Catania" ], [ 18393, "Enna" ], [ 18394, "Messina" ], [ 18395, "Palermo" ], [ 18396, "Ragusa" ], [ 18397, "Siracusa" ], [ 18398, "Trapani" ], [ 18399, "Arezzo" ], [ 18401, "Grosseto" ], [ 18402, "Livorno" ], [ 18403, "Lucca" ], [ 18404, "Massa-carrara" ], [ 18405, "Pisa" ], [ 18406, "Pistoia" ], [ 18408, "Siena" ], [ 18411, "Perugia" ], [ 18412, "Terni" ], [ 18335, "Ferrara" ], [ 18343, "Gorizia" ], [ 18344, "Pordenone" ], [ 18345, "Trieste" ], [ 18346, "Udine" ], [ 18356, "Bergamo" ], [ 18357, "Brescia" ], [ 18358, "Como" ], [ 18359, "Cremona" ], [ 18360, "Lecco" ], [ 18361, "Lodi" ], [ 18362, "Mantova" ], [ 18363, "Milano" ], [ 18364, "Pavia" ], [ 18365, "Sondrio" ], [ 18366, "Varese" ], [ 18375, "Biella" ], [ 18377, "Novara" ], [ 18379, "Verbania" ], [ 18380, "Vercelli" ], [ 18409, "Bolzano" ], [ 18410, "Trento" ], [ 18413, "Aosta" ], [ 18414, "Belluno" ], [ 18415, "Padova" ], [ 18416, "Rovigo" ], [ 18417, "Treviso" ], [ 18418, "Venezia" ], [ 18419, "Verona" ], [ 18420, "Vicenza" ] ]
mapping = {key: value for (key, value) in mapping}

def plot_province_errors(model, svg_file=Path("italy_provinces.svg"), output_file="italy_plot.svg"):
    """
    For the most part, reformatting of cyp/analysis/counties_plot.py

    Generates an svg of the Italian provinces, coloured by their prediction error.
    Additionally, prints statistics about the prediction errors for the input model, for train, test and italy validation sets

    The svg bar save code can be reused from the original plotting script (counties_plot.py).

    Parameters
    ----------
    model: pathlib Path
        Path to the model being plotted.
    svg_file: pathlib Path, default=("data/italy_provinces.svg")
        Path to the italy province svg used as a base
    """
    model_sd = torch.load(model, map_location="cpu")

    real_values, pred_values = model_sd["it_real"], model_sd["it_pred"]
    indices = model_sd["it_indices"]

    pred_err = pred_values - real_values
    pred_dict = {}
    for idx, err in zip(indices, pred_err):
        pred_dict[mapping[idx[1]]] = err

    it_stats = calculate_stats(pred_values, real_values)

    train_real_values, train_pred_values = model_sd["train_real"], model_sd["train_pred"]
    train_stats = calculate_stats(train_pred_values, train_real_values)

    test_real_values, test_pred_values = model_sd["test_real"], model_sd["test_pred"]
    test_stats = calculate_stats(test_pred_values, test_real_values)
    

    # Print stats about the model
    print(f"Model: {model}")
    print(f"Train stats: MSE: {train_stats['mse']}, MAE: {train_stats['mae']}, Min ABS Error: {train_stats['min_abs_err']}, Max ABS Error {train_stats['max_abs_err']}")
    print(f"Test stats: MSE: {test_stats['mse']}, MAE: {test_stats['mae']}, Min ABS Error: {test_stats['min_abs_err']}, Max ABS Error {test_stats['max_abs_err']}")
    print(f"Validation (Italy) stats: MSE {it_stats['mse']}, MAE: {it_stats['mae']}, Min ABS Error: {it_stats['min_abs_err']}, Max ABS Error {it_stats['max_abs_err']}")
    print("\n")

    colors = [
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
    ]

    _single_plot(
        pred_dict, svg_file, output_file, colors
    )


def _single_plot(err_dict, svg_file, savepath, colors):
    # load the svg file
    svg = svg_file.open("r").read()
    # Load into Beautiful Soup
    soup = BeautifulSoup(svg, features="html.parser")
    # Find provinces
    paths = soup.findAll("path")

    path_style = (
        "font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1"
        ";stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start"
        ":none;stroke-linejoin:bevel;fill:"
    )
    covered = []
    for p in paths:
        if p["name"] in err_dict:
            rate = err_dict[p["name"]]
            covered.append(p["name"])
            
            if rate > 15:
                color_class = 7
            elif rate > 10:
                color_class = 6
            elif rate > 5:
                color_class = 5
            elif rate > 0:
                color_class = 4
            elif rate > -5:
                color_class = 3
            elif rate > -10:
                color_class = 2
            elif rate > -15:
                color_class = 1
            else:
                color_class = 0

            color = colors[color_class]
            p["style"] = path_style + color
    for key in err_dict:
        if key not in covered: print(key)
    soup = soup.prettify()
    with open(savepath, "w") as f:
        f.write(soup)


def calculate_stats(pred_values, real_values):
    return {
        "mse": round(np.square(np.subtract(pred_values, real_values)).mean(), 5), 
        "mae": round(np.abs(np.subtract(pred_values, real_values)).mean(), 5), 
        "min_abs_err": round(np.min(np.abs(np.subtract(pred_values, real_values))), 5), 
        "max_abs_err": round(np.max(np.abs(np.subtract(pred_values, real_values))), 5)
    }