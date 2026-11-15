def test_import():
    import qjepa
    from qjepa.config import LATENT

    assert qjepa.__version__ == "0.1.0"
    assert LATENT == (0, 1)
