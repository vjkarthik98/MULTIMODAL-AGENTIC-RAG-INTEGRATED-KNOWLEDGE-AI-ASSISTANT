#Requires -Version 5.1
<#
Rebuild llama-cpp-python with Vulkan GPU offload for AMD Radeon 680M iGPU
(Ryzen 7 7735HS). Run from inside the rag_env conda environment.

Prereqs (script will check and abort if any are missing):
  1. Vulkan SDK installed:  https://vulkan.lunarg.com/sdk/home#windows
     Sets VULKAN_SDK env var.
  2. CMake on PATH:         winget install Kitware.CMake
  3. MSVC compiler on PATH: launch from "x64 Native Tools Command Prompt for VS"
                            OR run vcvarsall.bat x64 first in this shell.
  4. Active Python env:     conda activate rag_env

The script captures the current llama-cpp-python version before uninstalling
and will reinstall the CPU build on failure so your project keeps working.
#>

$ErrorActionPreference = 'Stop'

function Write-Section($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Fail($msg)    { Write-Host "[FAIL]  $msg" -ForegroundColor Red }
function Write-Warn($msg)    { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }

# ---------------------------------------------------------------------------
# Phase 1 - Prerequisite checks
# ---------------------------------------------------------------------------
Write-Section "Prerequisite checks"

$failed = @()

# Python env check - must be rag_env (sys.prefix contains 'rag_env')
$pyPrefix = (python -c "import sys; print(sys.prefix)") 2>$null
if (-not $pyPrefix) { $failed += "python not callable from this shell"; }
elseif ($pyPrefix -notmatch 'rag_env') {
    Write-Fail "Python prefix is '$pyPrefix' - expected to contain 'rag_env'."
    Write-Host "   Run: conda activate rag_env" -ForegroundColor Yellow
    $failed += "wrong python env"
} else { Write-Ok "Python env: $pyPrefix" }

# Vulkan SDK
if (-not $env:VULKAN_SDK) {
    Write-Fail "VULKAN_SDK env var not set."
    Write-Host "   Install:  https://vulkan.lunarg.com/sdk/home#windows" -ForegroundColor Yellow
    Write-Host "   Then open a NEW shell so VULKAN_SDK is exported."   -ForegroundColor Yellow
    $failed += "vulkan sdk"
} else {
    if (Test-Path (Join-Path $env:VULKAN_SDK 'Include\vulkan\vulkan.h')) {
        Write-Ok "Vulkan SDK: $env:VULKAN_SDK"
    } else {
        Write-Fail "VULKAN_SDK is set ($env:VULKAN_SDK) but vulkan.h not found under Include\vulkan."
        $failed += "vulkan sdk headers"
    }
}

# CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Fail "cmake not on PATH."
    Write-Host "   Install: winget install Kitware.CMake" -ForegroundColor Yellow
    $failed += "cmake"
} else { Write-Ok "cmake: $($cmake.Source)" }

# MSVC compiler (cl.exe)
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $cl) {
    Write-Fail "cl.exe (MSVC) not on PATH."
    Write-Host "   Launch this shell from: Start Menu -> 'x64 Native Tools Command Prompt for VS 2022'" -ForegroundColor Yellow
    Write-Host "   OR run: & 'C:\Program Files\Microsoft Visual Studio\2022\<edition>\VC\Auxiliary\Build\vcvars64.bat'" -ForegroundColor Yellow
    $failed += "msvc"
} else { Write-Ok "MSVC: $($cl.Source)" }

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Fail "Aborting - missing prerequisites: $($failed -join ', ')"
    Write-Host "Fix the items above and re-run this script. Your existing llama-cpp-python install is untouched." -ForegroundColor Yellow
    exit 1
}

# ---------------------------------------------------------------------------
# Phase 2 - Capture current version (rollback reference)
# ---------------------------------------------------------------------------
Write-Section "Capturing current llama-cpp-python version"

$currentVersion = (python -c "import llama_cpp; print(llama_cpp.__version__)") 2>$null
if ($LASTEXITCODE -ne 0 -or -not $currentVersion) {
    Write-Warn "llama-cpp-python not currently installed - nothing to roll back to."
    $currentVersion = $null
} else {
    Write-Ok "Current version: $currentVersion (will reinstall this CPU build if Vulkan build fails)"
}

# ---------------------------------------------------------------------------
# Phase 3 - Uninstall current build
# ---------------------------------------------------------------------------
Write-Section "Uninstalling current llama-cpp-python"
pip uninstall -y llama-cpp-python
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Uninstall returned non-zero (may have already been absent). Continuing."
}

# ---------------------------------------------------------------------------
# Phase 4 - Rebuild with Vulkan
# ---------------------------------------------------------------------------
Write-Section "Building llama-cpp-python with Vulkan backend"
Write-Host "This compiles from source and can take 5-15 minutes." -ForegroundColor Yellow

$env:CMAKE_ARGS  = "-DGGML_VULKAN=on"
$env:FORCE_CMAKE = "1"

pip install llama-cpp-python --no-cache-dir --verbose

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Vulkan build failed."
    if ($currentVersion) {
        Write-Warn "Rolling back to CPU build $currentVersion ..."
        Remove-Item Env:CMAKE_ARGS  -ErrorAction SilentlyContinue
        Remove-Item Env:FORCE_CMAKE -ErrorAction SilentlyContinue
        pip install "llama-cpp-python==$currentVersion" --no-cache-dir
        if ($LASTEXITCODE -eq 0) { Write-Ok "Rollback complete - CPU build restored." }
        else { Write-Fail "Rollback also failed. Reinstall manually: pip install llama-cpp-python==$currentVersion" }
    } else {
        Write-Warn "No previous version to roll back to. Install CPU build with: pip install llama-cpp-python"
    }
    exit 1
}

# ---------------------------------------------------------------------------
# Phase 5 - Verify Vulkan backend is active
# ---------------------------------------------------------------------------
Write-Section "Verifying Vulkan backend"

$verifyScript = @'
import llama_cpp
import sys

# llama-cpp-python exposes the underlying llama.cpp build flags via the
# llama_print_system_info function. A Vulkan-enabled build prints "Vulkan = 1".
try:
    info_fn = llama_cpp.llama_print_system_info
    raw = info_fn()
    info = raw.decode() if isinstance(raw, bytes) else str(raw)
except Exception as exc:
    print(f"FAIL: could not read llama.cpp system info: {exc}")
    sys.exit(2)

print(info)
if "Vulkan = 1" in info or "VULKAN = 1" in info:
    print("VULKAN_ACTIVE")
    sys.exit(0)
else:
    print("VULKAN_NOT_ACTIVE")
    sys.exit(3)
'@

$verifyOut = python -c $verifyScript
Write-Host $verifyOut
if ($LASTEXITCODE -eq 0) {
    Write-Ok "Vulkan backend is active in the new build."
} else {
    Write-Fail "Build installed but Vulkan flag was not detected."
    Write-Warn "The new library may still default to CPU. Inspect the output above before trusting GPU offload."
    exit 1
}

# ---------------------------------------------------------------------------
# Phase 6 - Next steps (no automatic .env edit - user controls that)
# ---------------------------------------------------------------------------
Write-Section "Done - next steps"
Write-Host ""
Write-Host "1. Edit .env and set LLM_GPU_LAYERS=20 to start." -ForegroundColor White
Write-Host "   The Radeon 680M shares system RAM; if you see OOM or instability," -ForegroundColor White
Write-Host "   drop to 16, then 12, then 8. Mistral-7B Q4_K_M has 32 layers total." -ForegroundColor White
Write-Host ""
Write-Host "2. Smoke test:  python -c `"from app.llm.gguf_model import GgufModel; m = GgufModel(); print(m.generate('Say hello.'))`"" -ForegroundColor White
Write-Host ""
Write-Host "3. Watch first-token latency in logs/app.log. Expect 2-4x improvement vs CPU-only." -ForegroundColor White
