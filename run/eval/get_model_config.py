#!/usr/bin/env python3
"""model_configs.yaml 조회 헬퍼 — 셸 스크립트가 config를 single source of truth로 쓰게 한다.

사용 예:
  # 그룹 전체를 셸 루프용으로 출력 (model|||gen_kwargs|||env_key)
  python run/eval/get_model_config.py --entries api_models
  python run/eval/get_model_config.py --entries calibration_reference api_models

  # 특정 모델의 특정 필드
  python run/eval/get_model_config.py Qwen3.5-27B --field gen_kwargs
  python run/eval/get_model_config.py claude-opus-4.8 --field model

  # 키 목록
  python run/eval/get_model_config.py --list
"""
import argparse
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "model_configs.yaml"
MODEL_GROUPS = ("calibration_reference", "api_models", "open_models")


def load():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def iter_models(cfg, groups):
    for group in groups:
        for key, entry in cfg.get(group, {}).items():
            yield group, key, entry


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("key", nargs="?", help="모델 키 (예: claude-opus-4.8, Qwen3.5-27B)")
    p.add_argument("--field", help="출력할 필드 (model/gen_kwargs/router/env_key/...)")
    p.add_argument("--entries", nargs="+", metavar="GROUP", choices=MODEL_GROUPS,
                   help="그룹의 전 모델을 'model|||gen_kwargs|||env_key' 형식으로 한 줄씩 출력")
    p.add_argument("--list", action="store_true", help="전체 모델 키 나열")
    args = p.parse_args()

    cfg = load()

    if args.list:
        for group, key, entry in iter_models(cfg, MODEL_GROUPS):
            print(f"{group:22s} {key:28s} {entry.get('model', '')}")
        return

    if args.entries:
        for _, _, entry in iter_models(cfg, args.entries):
            print(f"{entry['model']}|||{entry['gen_kwargs']}|||{entry.get('env_key', '')}")
        return

    if not args.key:
        p.error("모델 키, --entries, --list 중 하나는 필요합니다")

    for _, key, entry in iter_models(cfg, MODEL_GROUPS):
        if key == args.key:
            if args.field:
                value = entry.get(args.field)
                if value is None:
                    print(f"필드 없음: {args.field}", file=sys.stderr)
                    sys.exit(1)
                print(value)
            else:
                yaml.safe_dump({key: entry}, sys.stdout, allow_unicode=True, sort_keys=False)
            return

    print(f"모델 키 없음: {args.key} (--list로 확인)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
