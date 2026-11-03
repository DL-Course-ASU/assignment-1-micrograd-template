"""Build report macros and a transparent browser viewer for captured CLI logs."""
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts'
metrics = json.loads((ART/'metrics.json').read_text())
verification = json.loads((ART/'verification.json').read_text())
suites = ET.parse(ART/'pytest.xml').getroot()
cases = suites.findall('.//testcase')
assert len(cases) == 37
assert not suites.findall('.//failure') and not suites.findall('.//error')
macros = {'TestCount':str(len(cases)), 'EpochCount':str(metrics['epochs']),
          'GradientError':f"${verification['max_absolute_error']:.3g}$",
          'TestCorrect':str(round(metrics['final']['test_accuracy']*metrics['test_samples']))}
for phase,prefix in [('initial','Initial'),('final','Final')]:
    for key,suffix in [('objective','Objective'),('train_mse','TrainMSE'),('test_mse','TestMSE')]:
        macros[prefix+suffix] = f'{metrics[phase][key]:.5f}'
for phase,prefix in [('initial','Initial'),('final','')]:
    for key,suffix in [('train_accuracy','TrainAccuracy'),('test_accuracy','TestAccuracy')]:
        macros[prefix+suffix] = f'{100*metrics[phase][key]:.2f}'
macros['GradientError'] = r'$'+f"{verification['max_absolute_error']/1e-10:.3f}"+r'\times10^{-10}$'
(ROOT/'report/results.tex').write_text('\n'.join(
    '\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n')
test_log = (ART/'pytest.txt').read_text(encoding='utf-8-sig')
verify_log = (ART/'verification.txt').read_text(encoding='utf-8-sig')
page = '''<!doctype html><meta charset="utf-8"><title>MicroGrad verification evidence</title>
<style>body{background:#edf2f6;color:#17354a;margin:0;font:18px Segoe UI,sans-serif}
main{max-width:1100px;margin:30px auto;background:white;padding:30px 40px;border-radius:12px}
h1{font-size:30px;margin:0 0 8px}p{color:#456075}h2{font-size:21px;margin-top:26px}
pre{font:18px/1.65 Consolas,monospace;white-space:pre-wrap;background:#132536;color:#e4f6f3;padding:22px;border-radius:8px}
.badge{color:#14623f;background:#dcf4e6;display:inline-block;padding:8px 14px;border-radius:6px;font-weight:bold}
small{color:#5f7387}</style><main><h1>MicroGrad | Verification evidence</h1>
<p>Yousef Mohamed Mahmoud Helmy Ismail · 20P7954</p>
<div class="badge">37 / 37 local tests passed</div>
<h2>Captured pytest output</h2><pre>COMMAND: python -m pytest -q\nTEST_LOG</pre>
<h2>Independent analytic and numerical checks</h2><pre>COMMAND: python -m scripts.verify\nVERIFY_LOG</pre>
<small>Verbatim CLI transcripts rendered in a browser. This is not a native terminal-window capture.<br>
Raw transcripts: artifacts/pytest.txt and artifacts/verification.txt; structured results: artifacts/pytest.xml.</small></main>'''
page = page.replace('TEST_LOG',html.escape(test_log)).replace('VERIFY_LOG',html.escape(verify_log))
(ART/'verification.html').write_text(page,encoding='utf-8')
print('Report macros and evidence viewer generated.')
