#!/usr/bin/env python3
# pip install ultralytics pyyaml

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def _abs_source(path_root: Path, val_field):
    def fix_one(p):
        p = Path(p)
        return p if p.is_absolute() else (path_root / p).resolve()
    if isinstance(val_field, (list, tuple)):
        return [str(fix_one(p)) for p in val_field]
    return str(fix_one(val_field))

def main():
    parser = argparse.ArgumentParser(description="Валидация YOLO-модели и предсказания с сохранением JSON и картинок")
    parser.add_argument("--model", type=Path, required=True, help="Путь к .pt модели")
    parser.add_argument("--data", type=Path, required=True, help="Путь к data.yaml")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Каталог для результатов")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--seed", type=int, default=37, help="Случайное начальное число")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # загрузка и правка yaml
    data_path = args.data.resolve()
    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # всегда прописываем абсолютный path
    data_cfg["path"] = str(data_path.parent.resolve())

    tmp_yaml = out_dir / "_data_resolved.yaml"
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True)

    # Загрузка модели
    model = YOLO(str(args.model))

    # Валидация
    model.val(
        data=str(tmp_yaml),
        save_json=True,
        seed=int(args.seed),
        project=str(out_dir),
        name="val",
        exist_ok=True,
    )

    model.predict(
        source=data_cfg["path"] + '/images/val',
        conf=args.conf,
        save=True,
        seed=int(args.seed),
        project=str(out_dir),
        name="pred",
        exist_ok=True,
    )

    print(f"Результаты: {out_dir}")


if __name__ == "__main__":
    main()
