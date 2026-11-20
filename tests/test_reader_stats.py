import pandas as pd

from training.reader_stats import crop_geometry, leading_zero, profile


def _labels():
    return pd.DataFrame({
        "pseudonimo": ["F-a", "F-b", "F-c", "F-d"],
        "classe": ["display_digital", "display_ciclometrico", "numero_serie", "display_digital"],
        "x0": [0.0, 0.0, 0.0, 0.0], "y0": [0.0, 0.0, 0.0, 0.0],
        "x1": [100.0, 50.0, 80.0, 10.0], "y1": [20.0, 10.0, 10.0, 10.0],
        "status": ["aceito", "humano", "aceito", "discorda"],
        "texto": ["0123", "12[56]4", "AB12", "9"],
        "funcao": ["03", "", "", ""],
        "duas_leituras": ["True", "", "", ""],
        "rotacao": ["0", "90", "", "0"],
    })


def test_crop_geometry_aplica_margem_e_rotacao():
    geo = crop_geometry(_labels())
    # digital: 100 * (1 + 2 * 0,22) = 144; 20 * (1 + 2 * 0,27) = 30,8
    assert round(geo.loc[0, "largura"], 1) == 144.0
    assert round(geo.loc[0, "altura"], 1) == 30.8
    # ciclométrico girado 90 graus: largura e altura trocadas
    assert round(geo.loc[1, "altura"], 1) == 73.0
    assert geo.loc[2].isna().all()  # rotação desconhecida


def test_leading_zero():
    assert leading_zero(pd.Series(["0123", "0", "123"])).tolist() == [True, False, False]


def test_profile_conta_so_os_alvos():
    out = profile(_labels(), pd.Series({"F-a": "20052025", "F-b": "21052025"}))
    assert out["alvo_por_lote"].loc["All", "All"] == 3
    assert out["digital"].iloc[0].tolist() == [1, 1, 1, 0]
    assert out["ciclometrico"].iloc[0].tolist() == [1, 1]
    assert out["serie_letras"].iloc[0].tolist() == [1, 1]
    assert out["caracteres"].loc["display_ciclometrico", "50%"] == 4
