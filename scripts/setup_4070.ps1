# Restore the training sources at the path shared by the trainer and NecroPilot.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$upstream = 'https://github.com/hmty-CHEN/AlphaDefect-Public.git'
$revision = 'd5b3c384b4d36c8239d36b2b99e1e3b44a1c3dcb'
$external = Join-Path $root 'external'
$repo = Join-Path $external 'AlphaDefect-Public'
$cache = Join-Path $external 'alphadefect-upstream.git'
$stamp = Join-Path $repo 'UPSTREAM_COMMIT'
$patches = @('alphadefect-v1140-win.patch', 'alphadefect-v1140-solver-pool.patch', 'necropilot-room-save.patch')

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: git $args" }
}

if (!(Test-Path -LiteralPath $repo)) {
    $null = New-Item -ItemType Directory -Force -Path $external
    Invoke-Git init --bare --quiet $cache
    Invoke-Git -C $cache fetch --quiet --depth=1 $upstream $revision
    # Some upstream stats filenames contain colons. Only extract the Windows-compatible subtree.
    $archive = [IO.Path]::GetTempFileName()
    $stage = Join-Path $external ("alphadefect-setup-" + [guid]::NewGuid().ToString('N'))
    try {
        # Git for Windows validates unrelated paths while reading the root tree, even for archive.
        # Limit this override to writing a zip for the three allowlisted paths at this pinned commit.
        Invoke-Git -C $cache -c core.protectNTFS=false archive --format=zip "--output=$archive" $revision game-v0.114.0 LICENSE NOTICE.md
        $null = New-Item -ItemType Directory -Path $stage
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $stage)
        Set-Content -LiteralPath (Join-Path $stage 'UPSTREAM_COMMIT') -Value $revision -Encoding ascii
        # Stage and destination are siblings under the resolved external directory.
        [IO.Directory]::Move($stage, $repo)
    } finally {
        Remove-Item -LiteralPath $archive
    }
}
if (Test-Path -LiteralPath (Join-Path $repo '.git')) {
    $head = Invoke-Git -C $repo rev-parse HEAD
    $gitContext = @('-C', $repo)
} elseif (Test-Path -LiteralPath $stamp) {
    $head = Get-Content -LiteralPath $stamp -TotalCount 1
    $gitContext = @('-C', $repo, "--git-dir=$cache", "--work-tree=$repo")
} else {
    throw "Unrecognized source directory: $repo"
}
if ($head -ne $revision) {
    throw "Expected $revision at $repo, found $head. Use a fresh project checkout or review the existing checkout."
}
foreach ($name in $patches) {
    $patch = (Resolve-Path (Join-Path $root "patches/$name")).Path
    & git @gitContext apply --reverse --check --quiet $patch
    if ($LASTEXITCODE -ne 0) {
        Invoke-Git @gitContext apply --check $patch
        Invoke-Git @gitContext apply $patch
    }
}
Write-Host "AlphaDefect $revision is ready at $repo/game-v0.114.0"
