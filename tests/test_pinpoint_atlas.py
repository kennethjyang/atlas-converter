import unittest

from pinpoint_atlas import AtlasStructure, PinpointAtlas


def _structure() -> AtlasStructure:
    return AtlasStructure(
        name="Root",
        acronym="RT",
        parent_id=None,
        children_ids=set(),
    )


class PinpointAtlasTests(unittest.TestCase):
    def test_accepts_rgba_lut(self):
        atlas = PinpointAtlas(
            name="atlas",
            resolutions=[100],
            root_id=1,
            structures=[None, _structure()],
            lut=[0, 0, 0, 255, 10, 20, 30, 255],
        )

        self.assertEqual(atlas.lut, [0, 0, 0, 255, 10, 20, 30, 255])

    def test_rejects_rgb_lut(self):
        with self.assertRaises(ValueError):
            PinpointAtlas(
                name="atlas",
                resolutions=[100],
                root_id=1,
                structures=[None, _structure()],
                lut=[0, 0, 0],
            )

    def test_rejects_non_opaque_alpha(self):
        with self.assertRaises(ValueError):
            PinpointAtlas(
                name="atlas",
                resolutions=[100],
                root_id=1,
                structures=[None, _structure()],
                lut=[0, 0, 0, 128],
            )


if __name__ == "__main__":
    unittest.main()
