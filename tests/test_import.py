def test_import():
    import circuitjepa
    from circuitjepa.config import LATENT

    assert circuitjepa.__version__ == "0.1.0"
    assert LATENT == (0, 1)
