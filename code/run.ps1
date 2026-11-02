param([ValidateSet('serve','train','benchmark','verify','test')][string]$Mode='serve')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$python=Join-Path $PSScriptRoot 'python/.venv/Scripts/python.exe'
Set-Location $root
if(-not (Test-Path -LiteralPath $python)){throw '先按 code/README.md 创建 Python 虚拟环境并安装依赖'}
if($Mode -eq 'train'){
    & $python code/python/download_hydraulic.py
    if($LASTEXITCODE -ne 0){throw '数据下载失败'}
    foreach($task in @('valve','cooler','pump','accumulator')){
        & $python code/python/train_hydraulic.py --task $task
        if($LASTEXITCODE -ne 0){throw "训练失败: $task"}
    }
    # 保留整周期基线，再按验证集选择阀门窗口模型；部署种子不按测试效果挑选。
    $baseline=(Get-Content outputs/valve/latest.json -Raw | ConvertFrom-Json).run_id
    & $python code/python/study_valve.py --baseline-run-id $baseline
    if($LASTEXITCODE -ne 0){throw '阀门消融实验失败'}
    $study=(Get-Content outputs/valve-study/latest.json -Raw | ConvertFrom-Json).run_id
    $selected=Get-Content "outputs/valve-study/runs/$study/selection.json" -Raw | ConvertFrom-Json
    & $python code/python/train_hydraulic.py --task valve --profile valve-switch-100hz --epochs 80 --seed $selected.seed --architecture $selected.architecture --normal-policy $selected.normal_policy --selection-study-run $study
    if($LASTEXITCODE -ne 0){throw '阀门部署模型训练失败'}
    & $python code/python/experiment_tcn.py
    if($LASTEXITCODE -ne 0){throw 'TCN 实验失败'}
}elseif($Mode -eq 'benchmark'){
    # 原版 aeon/Numba 单独安装，避免其约束改变生产推理环境。
    $reference=Join-Path $root 'work/reference-env/Scripts/python.exe'
    if(-not (Test-Path -LiteralPath $reference)){throw '先按 code/README.md 建立 work/reference-env 并安装 requirements.reference.lock.txt'}
    foreach($dataset in @('cranfield-actuator','cwru','metropt3','cmapss','resnet18-weights')){
        & $python code/python/download_benchmarks.py $dataset
        if($LASTEXITCODE -ne 0){throw "下载失败: $dataset"}
    }
    & $python code/python/prepare_visa.py
    if($LASTEXITCODE -ne 0){throw 'VisA 子集准备失败'}
    foreach($arguments in @(
        @('code/python/study_actuator.py'),
        @('code/python/study_actuator.py','--reference-alignment'),
        @('code/python/study_actuator.py','--record-holdout'),
        @('code/python/study_actuator_ensemble.py'),
        @('code/python/study_rotating.py'),
        @('code/python/study_metro.py'),
        @('code/python/study_rul.py'),
        @('code/python/study_vision.py')
    )){
        & $python @arguments
        if($LASTEXITCODE -ne 0){throw "实验失败: $($arguments -join ' ')"}
    }
    & $reference code/python/study_reference_classifiers.py
    if($LASTEXITCODE -ne 0){throw '原版强基线失败'}
    $referenceRun=(Get-Content outputs/reference-study/latest.json -Raw | ConvertFrom-Json).run_id
    & $reference code/python/export_reference_parameters.py $referenceRun
    if($LASTEXITCODE -ne 0){throw 'MiniRocket 数值参数导出失败'}
    & $python code/python/export_minirocket.py $referenceRun
    if($LASTEXITCODE -ne 0){throw 'MiniRocket ONNX 校验失败'}
    & $python code/python/package_deployments.py
    if($LASTEXITCODE -ne 0){throw '统一部署包验证失败'}
}elseif($Mode -eq 'verify'){
    foreach($script in @('verify_artifacts.py','verify_runtime.py','verify_domains.py')){
        & $python "code/python/$script"
        if($LASTEXITCODE -ne 0){throw "验收失败: $script"}
    }
}elseif($Mode -eq 'test'){
    & $python -m pytest code/python/tests -q
    if($LASTEXITCODE -ne 0){throw 'Python 测试失败'}
    Push-Location code/ts/demo
    try{npm run build; if($LASTEXITCODE -ne 0){throw '前端构建失败'}}finally{Pop-Location}
}else{
    if(-not (Test-Path code/ts/demo/dist/index.html)){throw '先在 code/ts/demo 执行 npm ci 和 npm run build'}
    & $python code/python/serve_demo.py --port 8765
}
