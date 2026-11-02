"""Build and optionally install MAVIS's illumination-capable Windows binding locally."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

from setup_optitrack import REVISION, RUNTIME_FILES, check_runtime, validate_interpreter

SDK_SOURCE = 'examples/01_basic_tracking/unity_hand_tracking/optitrack_tools/code_base/Camera SDK/'
BINDING = 'optitrack_cam.cp311-win_amd64.pyd'


def restore_sdk(repo):
    destination = repo / 'external/optitrack-sdk-1.10/sdk'
    if not destination.resolve().is_relative_to(repo.resolve()):
        raise RuntimeError('SDK destination must remain inside the checkout')
    tree = subprocess.check_output(['git','ls-tree','-r','-l',REVISION,'--',
        SDK_SOURCE+'include',SDK_SOURCE+'lib/CameraLibrary2015x64S.lib'],cwd=repo,text=True)
    entries=[]
    for line in tree.splitlines():
        meta,source=line.split('\t',1)
        mode,kind,blob,size=meta.split()
        relative=PurePosixPath(source[len(SDK_SOURCE):])
        if kind!='blob' or relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Invalid SDK tree entry')
        entries.append((relative,blob,int(size)))
    if len(entries)!=188:
        raise RuntimeError(f'Expected 187 SDK headers and one import library; found {len(entries)}')
    objects=subprocess.check_output(['git','cat-file','--batch'],cwd=repo,
        input=''.join(blob+'\n' for _,blob,_ in entries).encode())
    offset=0
    for relative,blob,size in entries:
        end=objects.index(b'\n',offset)
        if objects[offset:end].decode().split()!=[blob,'blob',str(size)]:
            raise RuntimeError('Unexpected Git object header')
        data=objects[end+1:end+1+size]
        if hashlib.sha1(b'blob '+str(size).encode()+b'\0'+data).hexdigest()!=blob:
            raise RuntimeError('Git object verification failed')
        offset=end+size+2
        target=destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError('SDK path resolves outside its destination')
        if target.exists() and target.read_bytes()!=data:
            raise RuntimeError(f'Refusing to overwrite changed SDK file: {target}')
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
    return destination


def verify_binding(binding, runtime):
    code = """
import importlib, os, sys
with os.add_dll_directory(sys.argv[2]):
    sys.path.insert(0,sys.argv[1])
    sdk=importlib.import_module('optitrack_cam')
    assert sdk.BINDING_API_VERSION == 2
    for name in ('get_intensity','min_intensity','max_intensity','set_intensity',
                 'is_high_power_supported','is_high_power_available','get_high_power','set_high_power'):
        assert callable(getattr(sdk.Camera,name)),name
    print('PASS: native binding API v2 imports and exposes all illumination methods; no cameras initialized.')
"""
    subprocess.run([sys.executable,'-B','-c',code,str(binding.parent),str(runtime)],check=True)


def install_binding(binding, runtime, manifest):
    current=runtime/BINDING
    if current.is_symlink():
        raise RuntimeError('Refusing to replace a binding symlink')
    backup=runtime/(BINDING+'.legacy-backup')
    if not backup.exists():
        shutil.copy2(current,backup)
    temporary=runtime/(BINDING+'.new')
    shutil.copy2(binding,temporary)
    try:
        temporary.replace(current)
    except PermissionError as exc:
        raise RuntimeError('Close every MAVIS/Python process using this binding, then retry installation') from exc
    (runtime/'binding-build.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Installed local binding; original preserved as',backup)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-sdk',action='store_true',help='recover build headers/library from local Git history')
    parser.add_argument('--output',type=Path,default=Path('.cache/optitrack-binding'))
    parser.add_argument('--install',action='store_true',help='install locally after native import verification')
    parser.add_argument('--bundle',action='store_true',help='create a local binding-only ZIP; never uploads it')
    args=parser.parse_args(argv)
    validate_interpreter()
    repo=Path(__file__).resolve().parents[1]
    runtime=check_runtime(repo)
    if not args.restore_sdk:
        parser.error('Use --restore-sdk to verify the pinned SDK headers/import library before compiling')
    sdk=restore_sdk(repo)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    source=repo/'examples/01_basic_tracking/unity_hand_tracking/optitrack_tools/code_base'
    pybind_dir=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    build=output/'build'
    native=output/'runtime'
    subprocess.run(['cmake','-S',str(source),'-B',str(build),'-A','x64',
        f'-DCAMERA_SDK_DIR={sdk}',f'-DCAMERA_BINDING_OUTPUT_DIR={native}',
        f'-DPython3_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pybind_dir}'],check=True)
    subprocess.run(['cmake','--build',str(build),'--config','Release','--parallel','2'],check=True)
    binding=native/BINDING
    if not binding.is_file():
        raise RuntimeError('Expected CPython 3.11 x64 binding was not built')
    verify_binding(binding,runtime)
    manifest={'binding_api':2,'python_abi':'cp311-win_amd64',
        'binding_sha256':hashlib.sha256(binding.read_bytes()).hexdigest(),
        'sdk_revision':REVISION,'sdk_dll_sha256':RUNTIME_FILES['CameraLibrary2015x64S.dll']['sha256'],
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()}
    (output/'binding-build.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if args.install:
        install_binding(binding,runtime,manifest)
    if args.bundle:
        dist=importlib.metadata.distribution('pybind11')
        licenses=[dist.locate_file(item) for item in (dist.files or [])
                  if str(item).replace('\\','/').endswith('/LICENSE')]
        if not licenses:
            raise RuntimeError('pybind11 license file not found')
        bundle=output/'optitrack-binding-cp311-win_amd64.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(binding,BINDING)
            archive.writestr('binding-build.json',json.dumps(manifest,indent=2)+'\n')
            archive.write(repo/'LICENSE','LICENSE')
            archive.write(repo/'THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md')
            archive.write(licenses[0],'PYBIND11_LICENSE.txt')
        print('Local bundle:',bundle)
        print('Bundle SHA256:',hashlib.sha256(bundle.read_bytes()).hexdigest())
    print('Binding SHA256:',manifest['binding_sha256'])
    print('No vendor DLLs, headers or libraries were bundled or uploaded.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
