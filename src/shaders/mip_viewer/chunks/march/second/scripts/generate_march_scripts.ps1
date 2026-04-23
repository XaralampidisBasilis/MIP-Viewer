$ErrorActionPreference = "Stop"

$marchDir = Split-Path -Parent $PSScriptRoot
$scriptsDir = $PSScriptRoot

function Resolve-IncludePath {
    param(
        [string]$BaseFile,
        [string]$IncludeTarget
    )

    $candidate = Join-Path (Split-Path -Parent $BaseFile) $IncludeTarget

    if (-not [System.IO.Path]::HasExtension($candidate)) {
        $candidate = "$candidate.glsl"
    }

    return [System.IO.Path]::GetFullPath($candidate)
}

function Expand-Includes {
    param(
        [string]$FilePath
    )

    $lines = Get-Content -LiteralPath $FilePath
    $expandedLines = New-Object System.Collections.Generic.List[string]

    foreach ($line in $lines) {
        if ($line -match '^(?<indent>\s*)#include\s+"(?<target>[^"]+)"\s*$') {
            $includePath = Resolve-IncludePath -BaseFile $FilePath -IncludeTarget $matches.target
            $expanded = Expand-Includes -FilePath $includePath

            foreach ($expandedLine in ($expanded -split "`r?`n")) {
                $expandedLines.Add("$($matches.indent)$expandedLine")
            }
        }
        else {
            $expandedLines.Add($line)
        }
    }

    return ($expandedLines -join [Environment]::NewLine)
}

foreach ($sourceDir in @($marchDir, $scriptsDir)) {
    $resultsDir = Join-Path $sourceDir "results"
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

    $sourceFiles = Get-ChildItem -LiteralPath $sourceDir -Filter "*.glsl" -File |
        Where-Object {
            $_.BaseName -like "march_cells*" -or $_.BaseName -like "march_traces*"
        }

    foreach ($sourceFile in $sourceFiles) {
        $expandedContent = Expand-Includes -FilePath $sourceFile.FullName
        $outputPath = Join-Path $resultsDir $sourceFile.Name
        Set-Content -LiteralPath $outputPath -Value $expandedContent
    }
}

$variantSourceFiles = foreach ($sourceDir in @(
    (Join-Path $marchDir "first"),
    (Join-Path $marchDir "second")
)) {
    Get-ChildItem -LiteralPath $sourceDir -Filter "*.glsl" -File |
        Where-Object {
            $_.Name -notlike "*_march.glsl" -and
            $_.Name -notlike "*_march_cells.glsl" -and
            $_.Name -notlike "*_march_traces.glsl"
        }
}

foreach ($sourceFile in $variantSourceFiles) {
    $expandedContent = Expand-Includes -FilePath $sourceFile.FullName
    $outputPath = Join-Path $scriptsDir $sourceFile.Name
    Set-Content -LiteralPath $outputPath -Value $expandedContent
}
