from __future__ import annotations

from scripts.list_hf_collide_dirs import infer_hf_data_dirs, list_hf_data_dirs


class FakeHfApi:
    def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
        assert repo_id == "fastmachinelearning/collide-1m"
        assert repo_type == "dataset"
        return [
            "README.md",
            "WJetsToLNu_13TeV-madgraphMLM-pythia8/part-00000.parquet",
            "TT_TuneCP5_13TeV-powheg-pythia8/part-00000.parquet",
            "TT_TuneCP5_13TeV-powheg-pythia8/notes.txt",
            "flat.parquet",
            "ZJetsToNuNu_HT-100To200_13TeV-madgraph/part-00001.parquet",
        ]


def test_infer_hf_data_dirs_from_parquet_paths() -> None:
    data_dirs = infer_hf_data_dirs(
        [
            "proc_b/file.parquet",
            "proc_a/file.parquet",
            "proc_a/other.parquet",
            "proc_c/readme.txt",
            "root.parquet",
        ]
    )

    assert data_dirs == ["proc_a", "proc_b"]


def test_list_hf_data_dirs_uses_repo_file_metadata_only() -> None:
    data_dirs = list_hf_data_dirs(
        "fastmachinelearning/collide-1m",
        max_dirs=2,
        api=FakeHfApi(),
    )

    assert data_dirs == [
        "TT_TuneCP5_13TeV-powheg-pythia8",
        "WJetsToLNu_13TeV-madgraphMLM-pythia8",
    ]
