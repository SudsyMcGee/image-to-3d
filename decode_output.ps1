<#
.SYNOPSIS
    Decode a base64 mesh from a RunPod JSON response saved to a file.

.EXAMPLE
    # Save the curl/Invoke-RestMethod response body to response.json first, then:
    .\decode_output.ps1 response.json
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ResponseFile
)

$resp = Get-Content $ResponseFile -Raw | ConvertFrom-Json
$b64  = $resp.output.mesh_b64

if (-not $b64) {
    Write-Error "No mesh_b64 field found in response."
    exit 1
}

$fmt     = if ($resp.output.format) { $resp.output.format } else { "glb" }
$outFile = "decoded_mesh.$fmt"

[System.IO.File]::WriteAllBytes(
    (Join-Path $PWD $outFile),
    [System.Convert]::FromBase64String($b64)
)

Write-Host "Saved $outFile"
Write-Host "Vertices : $($resp.output.vertex_count)"
Write-Host "Faces    : $($resp.output.face_count)"
