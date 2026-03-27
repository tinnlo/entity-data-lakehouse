from entity_data_lakehouse.utils import normalize_name, stable_id


def test_normalize_name_strips_accents_and_spacing() -> None:
    assert normalize_name("Energías   Renovables, S.A.") == "energias renovables s a"


def test_stable_id_is_deterministic() -> None:
    assert stable_id("ent", "abc", 123) == stable_id("ent", "abc", 123)
