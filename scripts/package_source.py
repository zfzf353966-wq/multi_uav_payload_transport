#!/usr/bin/env python3
"""打包自研源码、当前策略和精选成果；不包含本机环境及私人申报表。"""
from __future__ import annotations
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# 显式白名单：不会意外收入 .git、虚拟环境、第三方仓库或旧运行目录。
FILES = ['README.md', '.gitignore', '.gitattributes', 'requirements.txt', 'requirements-docs.txt', 'dependencies.repos']
DIRECTORIES = ['src/logistics_planner', 'scripts', 'config', 'simulation', 'models/ppo_delivery', 'artifacts', 'docs']
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.git', '.venv', 'CMakeFiles'}


def source_files():
    selected = [ROOT / name for name in FILES]
    for directory in DIRECTORIES:
        selected.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    for path in sorted(set(selected)):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {'.pyc', '.pyo', '.log', '.so', '.o'} or path.name.startswith('~$'):
            continue
        if path.is_symlink():
            raise ValueError(f'交付目录不应包含指向外部的符号链接: {relative}')
        yield path, relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist')
    parser.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'三无人机城市配送_源程序与使用说明_{args.date}.zip'
    if output.exists():
        parser.error('输出文件已存在，请选择新目录，避免覆盖已提交版本。')
    entries = list(source_files())
    required = ['models/ppo_delivery/delivery_model.zip', 'docs/软件使用说明.doc', 'docs/软件使用说明.pdf']
    names = {relative for _, relative in entries}
    if not set(required) <= names:
        parser.error('请先准备当前模型并生成DOC/PDF使用说明。')
    checksums = {relative: hashlib.sha256(path.read_bytes()).hexdigest() for path, relative in entries}
    prefix = 'multi_uav_payload_transport/'
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in entries:
            archive.write(path, prefix + relative)
        archive.writestr(prefix + 'SOURCE_MANIFEST.json', json.dumps(checksums, ensure_ascii=False, indent=2))
        archive.writestr('提交说明.txt', '本包为三无人机城市配送与PPO动态避障源程序交付。\n'
                          '先打开 multi_uav_payload_transport/docs/软件使用说明.doc。\n'
                          '源码及必要注释在src/、simulation/plugins/和scripts/，当前模型在models/。\n'
                          '已配置电脑：项目根目录运行 scripts/run_ppo_delivery.bash --gui --seed 1。\n'
                          '新电脑须按使用说明安装Ubuntu、ROS、Gazebo、PX4及Python依赖。\n'
                          'SOURCE_MANIFEST.json可逐文件复核；不含本机环境、Git历史及私人申报表。\n')
    # 打包后读取每一项并验证校验，防止缺文件、截断或误打包。
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP完整性校验失败')
        for relative, expected in checksums.items():
            if hashlib.sha256(archive.read(prefix + relative)).hexdigest() != expected:
                raise RuntimeError(f'校验失败: {relative}')
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n')
    print(json.dumps({'file': str(output), 'source_files': len(entries), 'bytes': output.stat().st_size, 'sha256': checksum}, ensure_ascii=False))


if __name__ == '__main__':
    main()
