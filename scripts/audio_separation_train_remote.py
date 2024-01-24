import kaggle
from pathlib import Path, PurePosixPath
import json
from __kaggle_login import kaggle_users
import argparse
import sys
import subprocess
from gyraudio import root_dir as ROOT_DIR


def get_git_branch_name():
    try:
        branch_name = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()
        return branch_name
    except subprocess.CalledProcessError:
        return "Error: Could not determine the Git branch name."


def prepare_notebook(
    output_nb_path: Path,
    exp: int,
    branch: str,
    git_user: str = "balthazarneveu",
    git_repo: str = "gyraudio",
    template_nb_path: Path = Path(__file__).parent/"remote_training_template.ipynb"
):
    expressions = [
        ("exp", f"{exp}"),
        ("branch", f"\'{branch}\'"),
        ("git_user", f"\'{git_user}\'"),
        ("git_repo", f"\'{git_repo}\'")
    ]
    with open(template_nb_path) as f:
        template_nb = f.readlines()
        for line_idx, li in enumerate(template_nb):
            for expr, expr_replace in expressions:
                if f"!!!{expr}!!!" in li:
                    template_nb[line_idx] = template_nb[line_idx].replace(f"!!!{expr}!!!", expr_replace)
        template_nb = "".join(template_nb)
    with open(output_nb_path, "w") as w:
        w.write(template_nb)


def main(argv):
    nb_id = "train-audio-separation"

    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="Kaggle user", choices=list(kaggle_users.keys()))
    parser.add_argument("-e", "--exp", type=int, required=True, help="Experiment id")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--branch", type=str, help="Git branch name", default=get_git_branch_name())
    parser.add_argument("-p", "--push", action="store_true", help="Push")
    parser.add_argument("-d", "--download", action="store_true", help="Download results")
    args = parser.parse_args(argv)
    exp = args.exp
    kaggle_user = kaggle_users[args.user]
    uname_kaggle = kaggle_user["username"]
    kaggle.api._load_config(kaggle_user)
    if args.download:
        tmp_dir = ROOT_DIR/f"__tmp_{exp:04d}"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        kaggle.api.kernels_output_cli(f"{kaggle_user['username']}/{nb_id}", path=str(tmp_dir))
        subprocess.run(["tar", "-xzf", tmp_dir/"output.tgz", "__output_audiosep"])
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    kernel_root = ROOT_DIR/f"__nb_{uname_kaggle}"
    kernel_root.mkdir(exist_ok=True, parents=True)

    kernel_path = kernel_root/f"{exp:04d}"
    kernel_path.mkdir(exist_ok=True, parents=True)
    branch = args.branch
    config = {
        "id": str(PurePosixPath(f"{kaggle_user['username']}")/nb_id),
        "title": nb_id.lower(),
        "code_file": f"{nb_id}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if not args.cpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["balthazarneveu/audio-separation-dataset"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    prepare_notebook((kernel_path/nb_id).with_suffix(".ipynb"), exp, branch)
    assert (kernel_path/nb_id).with_suffix(".ipynb").exists()
    with open(kernel_path/"kernel-metadata.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.push:
        kaggle.api.kernels_push_cli(str(kernel_path))


if __name__ == '__main__':
    main(sys.argv[1:])
