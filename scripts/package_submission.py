"""Package tracked, committed artifacts; verify every archived byte by SHA-256."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'Yousef-Mohamed-Mahmoud-Helmy-Ismail_20P7954'
def git(*args):
    return subprocess.check_output(['git','-c','core.excludesfile=',*args],cwd=ROOT)


if __name__ == '__main__':
    assert git('branch','--show-current').decode().strip() == BRANCH
    assert not git('status','--porcelain').strip(), 'Commit all submission files first'
    tracked = git('ls-files','-z').decode().split('\0')
    files = [ROOT/name for name in tracked if name]
    required = ['src/engine.py','src/nn.py','notebooks/demo.ipynb',
                'report/Documentation_Report.pdf','report/report.tex',
                'report/figures/verification_screenshot.jpg',
                'report/figures/notebook_screenshot.jpg',
                'report/figures/notebook_results_screenshot.jpg']
    assert all(ROOT/name in files for name in required)
    manifest = {'student':'Yousef Mohamed Mahmoud Helmy Ismail','student_id':'20P7954',
                'branch':BRANCH,'commit':git('rev-parse','HEAD').decode().strip(),
                'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files}}
    manifest_bytes = json.dumps(manifest,indent=2).encode()
    (ROOT/'SUBMISSION_MANIFEST.json').write_bytes(manifest_bytes)
    archive = ROOT.parent/(BRANCH+'_Assignment1.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('SUBMISSION_MANIFEST.json',manifest_bytes)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)+1
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
        notebook = json.loads(z.read('notebooks/demo.ipynb'))
        cells = [c for c in notebook['cells'] if c['cell_type']=='code']
        assert [c['execution_count'] for c in cells] == list(range(1,len(cells)+1))
        assert not any(o['output_type']=='error' for c in cells for o in c['outputs'])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix('.zip.sha256')
    checksum.write_text(digest+'  '+archive.name+'\n')
    print(f'Archive: {archive}')
    print(f'Files: {len(files)+1} | Bytes: {archive.stat().st_size}')
    print(f'Commit: {manifest["commit"]}')
    print(f'SHA-256: {digest}')
    print('ZIP CRC, per-file hashes, notebook execution and required artifacts: PASS')
