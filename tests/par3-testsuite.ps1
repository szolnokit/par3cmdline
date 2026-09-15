# PAR3 end-to-end test suite (Windows PowerShell 5+)
# Usage: powershell -File tests\par3-testsuite.ps1 [-Exe path\to\par3.exe]
param(
    [string]$Exe = "$PSScriptRoot\..\src\build\par3cmd\Release\par3.exe",
    [string]$Work = "$env:TEMP\par3-testsuite"
)

$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$script:passed = 0
$script:failed = 0

function Report($name, $ok, $detail = "") {
    if ($ok) { $script:passed++; Write-Host "PASS  $name" -ForegroundColor Green }
    else     { $script:failed++; Write-Host "FAIL  $name  $detail" -ForegroundColor Red }
}

function New-RandFile($path, $size, $seed) {
    $rng = [System.Random]::new($seed)
    $b = New-Object byte[] $size
    $rng.NextBytes($b)
    [IO.File]::WriteAllBytes($path, $b)
}

function Damage($path, $offset, $len) {
    $bytes = [IO.File]::ReadAllBytes($path)
    for ($i = 0; $i -lt $len; $i++) { $bytes[$offset + $i] = $bytes[$offset + $i] -bxor 0xFF }
    [IO.File]::WriteAllBytes($path, $bytes)
}

function Hash($path) { (Get-FileHash $path -Algorithm SHA256).Hash }

function New-TestDir($name) {
    $dir = Join-Path $Work $name
    Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path "$dir\data" -Force | Out-Null
    return $dir
}

# --- Test 1-4: create / damage / repair for each ECC mode -------------------
$modes = @(
    @{ name = "cauchy-gf8";  args = @("-s1000", "-r40");        files = 20000 },
    @{ name = "cauchy-gf16"; args = @("-s100",  "-r30");        files = 40000 },
    @{ name = "sparse-gf16"; args = @("-s100",  "-r30", "-e2"); files = 40000 },
    @{ name = "fft";         args = @("-s1000", "-r40", "-e8"); files = 20000 }
)
foreach ($m in $modes) {
    $dir = New-TestDir $m.name
    New-RandFile "$dir\data\a.bin" $m.files 11
    New-RandFile "$dir\data\b.bin" ([int]($m.files / 2)) 12
    $hashA = Hash "$dir\data\a.bin"
    Push-Location "$dir\data"
    try {
        & $Exe c @($m.args) ..\test.par3 * *> $null
        $createOk = ($LASTEXITCODE -eq 0)
        Damage "$dir\data\a.bin" 100 ([int]($m.files / 10))
        & $Exe r ..\test.par3 *> $null
        $repairOk = ($LASTEXITCODE -eq 0) -and ((Hash "$dir\data\a.bin") -eq $hashA)
        Report "$($m.name): create+repair" ($createOk -and $repairOk)
    } finally { Pop-Location }
}

# --- Test 5: whole file loss ------------------------------------------------
$dir = New-TestDir "file-loss"
New-RandFile "$dir\data\a.bin" 5000 21
New-RandFile "$dir\data\b.bin" 5000 22
$hashB = Hash "$dir\data\b.bin"
Push-Location "$dir\data"
try {
    & $Exe c -s1000 -r100 ..\test.par3 * *> $null
    Remove-Item b.bin
    & $Exe r ..\test.par3 *> $null
    Report "file-loss: restore deleted file" (($LASTEXITCODE -eq 0) -and (Test-Path b.bin) -and ((Hash "$dir\data\b.bin") -eq $hashB))
} finally { Pop-Location }

# --- Test 5b: whole file loss with sparse codes (exercises the dense core of
# the peeling decoder: a contiguous loss leaves no degree-1 equations) --------
$dir = New-TestDir "file-loss-sparse"
New-RandFile "$dir\data\a.bin" 300000 23
New-RandFile "$dir\data\b.bin" 250000 24
$hashB = Hash "$dir\data\b.bin"
Push-Location "$dir\data"
try {
    & $Exe c -s1000 -r100 -e2 ..\test.par3 * *> $null
    Remove-Item b.bin
    $out = & $Exe r -v ..\test.par3 2>&1
    $peel = ($out -match "Peeling decoder")
    Report "file-loss-sparse: restore deleted file via peeling core" (($LASTEXITCODE -eq 0) -and $peel -and (Test-Path b.bin) -and ((Hash "$dir\data\b.bin") -eq $hashB))
} finally { Pop-Location }

# --- Test 6: intact verify returns 0 ----------------------------------------
$dir = New-TestDir "intact"
New-RandFile "$dir\data\a.bin" 3000 31
Push-Location "$dir\data"
try {
    & $Exe c -s500 -r20 ..\test.par3 * *> $null
    & $Exe v ..\test.par3 *> $null
    Report "intact: verify exit 0" ($LASTEXITCODE -eq 0)
} finally { Pop-Location }

# --- Test 7: incremental backup chain ----------------------------------------
$dir = New-TestDir "incremental"
New-RandFile "$dir\data\old_a.bin" 6000 41
New-RandFile "$dir\data\old_b.bin" 4000 42
Push-Location "$dir\data"
try {
    & $Exe c -s1000 -r40 ..\gen1.par3 * *> $null
    $g1 = ($LASTEXITCODE -eq 0)

    # additive child: new file only
    New-RandFile "$dir\data\new_c.bin" 5000 43
    $out = & $Exe c -s1000 -r40 "-P..\gen1.par3" ..\gen2.par3 * 2>&1
    $g2 = ($LASTEXITCODE -eq 0) -and ($out -match "Inherited 2 of 3")

    # child verifies whole tree
    & $Exe v ..\gen2.par3 *> $null
    $g3 = ($LASTEXITCODE -eq 0)

    # damage new file -> child repairs
    $hashC = Hash "$dir\data\new_c.bin"
    Damage "$dir\data\new_c.bin" 500 1500
    & $Exe r ..\gen2.par3 *> $null
    $g4 = ($LASTEXITCODE -eq 0) -and ((Hash "$dir\data\new_c.bin") -eq $hashC)

    # damage old file -> child reports covered-range failure, parent repairs
    $hashA = Hash "$dir\data\old_a.bin"
    Damage "$dir\data\old_a.bin" 200 1500
    $out = & $Exe r ..\gen2.par3 2>&1
    $g5 = ($LASTEXITCODE -ne 0) -and ($out -match "outside the range")
    & $Exe r ..\gen1.par3 *> $null
    $g6 = ($LASTEXITCODE -eq 0) -and ((Hash "$dir\data\old_a.bin") -eq $hashA)

    Report "incremental: parent create"          $g1
    Report "incremental: child create (-P)"      $g2
    Report "incremental: child verifies all"     $g3
    Report "incremental: child repairs new file" $g4
    Report "incremental: old damage -> clear message" $g5
    Report "incremental: parent repairs old file"     $g6
} finally { Pop-Location }

# --- Test 8: extended block count (PAR SPX, -e2x) -----------------------------
$dir = New-TestDir "sparse-extended"
New-RandFile "$dir\data\big.bin" 8000000 51    # 8 MB with -b80000 -> 100-byte blocks
Push-Location "$dir\data"
try {
    # 80000 blocks exceed the 16-bit field: plain -e2 must refuse with a hint
    $out = & $Exe c -b80000 -r5 -e2 ..\neg.par3 big.bin 2>&1
    Report "sparse-ext: -e2 above 64k rejected with hint" (($LASTEXITCODE -ne 0) -and ($out -match "e2x"))

    $hashBig = Hash "$dir\data\big.bin"
    & $Exe c -b80000 -r5 -e2x ..\spx.par3 big.bin *> $null
    $createOk = ($LASTEXITCODE -eq 0)
    # the index file must carry the PAR SPX packet type
    $spxOk = ([Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes("$dir\spx.par3"))).Contains("PAR SPX")
    Report "sparse-ext: create 80000 blocks (SPX packet)" ($createOk -and $spxOk)

    # damage 200 scattered blocks, then repair and compare hashes
    $bytes = [IO.File]::ReadAllBytes("$dir\data\big.bin")
    foreach ($block in (0..79999 | Get-Random -Count 200 -SetSeed 52)) {
        $off = $block * 100 + 3
        for ($j = 0; $j -lt 16; $j++) { $bytes[$off + $j] = $bytes[$off + $j] -bxor 0xFF }
    }
    [IO.File]::WriteAllBytes("$dir\data\big.bin", $bytes)
    & $Exe r ..\spx.par3 big.bin *> $null
    Report "sparse-ext: repair 200 damaged blocks" (($LASTEXITCODE -eq 0) -and ((Hash "$dir\data\big.bin") -eq $hashBig))
} finally { Pop-Location }

# --- Summary -----------------------------------------------------------------
Write-Host ""
Write-Host "Passed: $script:passed, Failed: $script:failed"
if ($script:failed -gt 0) { exit 1 } else { exit 0 }
