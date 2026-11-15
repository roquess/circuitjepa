def test_import():
    import circuitjepa
    from circuitjepa.config import LATENT

    assert circuitjepa.__version__ == "1.0.0"
    assert LATENT == (0, 1)
