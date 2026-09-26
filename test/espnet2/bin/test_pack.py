from argparse import ArgumentParser
from inspect import signature

import pytest

from espnet2.bin.cls_inference import Classification
from espnet2.bin.pack import (
    ClassificationPackedContents,
    RSTPackedContents,
    get_parser,
    main,
)
from espnet2.bin.rst_inference import get_parser as rst_inference_parser


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def test_cls_packing():
    packing_obj = ClassificationPackedContents()
    valid_args = signature(Classification.__init__).parameters
    for filename in packing_obj.files:
        assert filename in valid_args, f"{filename} not in {valid_args}"
    for yaml_filename in packing_obj.yaml_files:
        assert yaml_filename in valid_args, f"{yaml_filename} not in {valid_args}"


def test_rst_packing():
    # rst_inference has no class: its parser's options are the contract
    valid_args = {a.dest for a in rst_inference_parser()._actions}
    packing_obj = RSTPackedContents()
    for filename in packing_obj.files:
        assert filename in valid_args, f"{filename} not in {valid_args}"
    for yaml_filename in packing_obj.yaml_files:
        assert yaml_filename in valid_args, f"{yaml_filename} not in {valid_args}"


def test_rst_pack_and_unpack(tmp_path):
    import yaml

    from espnet2.main_funcs.pack_funcs import unpack

    files = {}
    for name in ("config.yaml", "voc_config.yaml"):
        (tmp_path / name).write_text(yaml.safe_dump({"x": 1}))
    for name in ("model.pth", "vocoder.pth"):
        (tmp_path / name).write_bytes(b"0")
    outpath = tmp_path / "packed.zip"
    main(
        [
            "rst",
            "--train_config",
            str(tmp_path / "config.yaml"),
            "--model_file",
            str(tmp_path / "model.pth"),
            "--vocoder_train_config",
            str(tmp_path / "voc_config.yaml"),
            "--vocoder_model_file",
            str(tmp_path / "vocoder.pth"),
            "--outpath",
            str(outpath),
        ]
    )
    files = unpack(str(outpath), str(tmp_path / "unpacked"))
    assert set(files) == {
        "train_config",
        "model_file",
        "vocoder_train_config",
        "vocoder_model_file",
    }
