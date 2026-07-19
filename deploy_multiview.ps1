<#
.SYNOPSIS
    Build, push, and deploy the Zero123++ multiview serverless endpoint on RunPod.

.DESCRIPTION
    .\deploy_multiview.ps1 build    — build and push the Docker image
    .\deploy_multiview.ps1 deploy   — create or update the RunPod serverless endpoint
    .\deploy_multiview.ps1 status   — list all serverless endpoints
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("build", "deploy", "status")]
    [string]$Command
)

$REGISTRY      = "ghcr.io"
$DOCKER_USER   = "sudsymcgee"
$IMAGE_NAME    = "zero123plus"
$IMAGE_TAG     = "latest"
$FULL_IMAGE    = "${REGISTRY}/${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

$ENDPOINT_NAME = "zero123plus"
$GPU_IDS       = "NVIDIA GeForce RTX 4090"
$MIN_WORKERS   = 0
$MAX_WORKERS   = 3

if (-not $env:RUNPOD_API_KEY) {
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^RUNPOD_API_KEY=(.+)$") { $env:RUNPOD_API_KEY = $Matches[1] }
        }
    }
}
if (-not $env:RUNPOD_API_KEY) {
    Write-Error "RUNPOD_API_KEY not set."
    exit 1
}

switch ($Command) {

    "build" {
        Write-Host "`n==> Building $FULL_IMAGE ..."
        docker build -f Dockerfile.multiview -t $FULL_IMAGE .
        if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed"; exit 1 }

        Write-Host "`n==> Pushing $FULL_IMAGE ..."
        docker push $FULL_IMAGE
        if ($LASTEXITCODE -ne 0) { Write-Error "Docker push failed"; exit 1 }

        Write-Host "`n==> Done. Image: $FULL_IMAGE"
    }

    "deploy" {
        Write-Host "`n==> Checking for existing endpoint '$ENDPOINT_NAME' ..."
        $existing = runpodctl serverless list --output=json 2>$null | ConvertFrom-Json |
                    Where-Object { $_.name -eq $ENDPOINT_NAME } |
                    Select-Object -First 1

        if ($existing) {
            $id = $existing.id
            Write-Host "Found endpoint $id — updating image ..."
            runpodctl serverless update `
                --id $id `
                --image-name $FULL_IMAGE
        } else {
            Write-Host "Creating endpoint '$ENDPOINT_NAME' ..."
            runpodctl serverless create `
                --name $ENDPOINT_NAME `
                --image-name $FULL_IMAGE `
                --gpu-ids "$GPU_IDS" `
                --workers-min $MIN_WORKERS `
                --workers-max $MAX_WORKERS `
                --env "HF_HOME=/workspace/hf_cache"
        }

        Write-Host "`n==> Endpoint list:"
        runpodctl serverless list --output=table
    }

    "status" {
        runpodctl serverless list --output=table
    }
}
