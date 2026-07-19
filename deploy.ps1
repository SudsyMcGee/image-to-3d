<#
.SYNOPSIS
    Build, push, and deploy the TRELLIS image-to-3D serverless endpoint on RunPod.

.DESCRIPTION
    Three subcommands:
      .\deploy.ps1 build    — build and push the Docker image
      .\deploy.ps1 deploy   — create or update the RunPod serverless endpoint
      .\deploy.ps1 test     — send test_input.json to the endpoint and stream output
      .\deploy.ps1 pod      — spin up an interactive on-demand pod for validation
      .\deploy.ps1 status   — list all serverless endpoints

.NOTES
    Prerequisites:
      - Docker Desktop running
      - runpodctl installed and configured (runpodctl config --apiKey <KEY>)
      - RUNPOD_API_KEY in environment (or .env file)
      - Logged into ghcr.io (gh auth token | docker login ghcr.io -u sudsymcgee --password-stdin)
#>

param(
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("build", "deploy", "test", "pod", "status")]
    [string]$Command
)

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------
$REGISTRY     = "ghcr.io"
$DOCKER_USER  = "sudsymcgee"
$IMAGE_NAME   = "trellis2-stl"
$IMAGE_TAG    = "latest"
$FULL_IMAGE   = "${REGISTRY}/${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

$ENDPOINT_NAME = "trellis2-stl"
$GPU_IDS       = "NVIDIA GeForce RTX 4090"       # 24 GB — cheapest card that fits TRELLIS
$MIN_WORKERS   = 0                                # scale-to-zero when idle
$MAX_WORKERS   = 3                                # cap concurrency

# Model to bake into the image (override for TRELLIS 2 when released on HF)
$MODEL_ID = "JeffreyXiang/TRELLIS-image-large"

# RunPod API key — read from environment or .env
if (-not $env:RUNPOD_API_KEY) {
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^RUNPOD_API_KEY=(.+)$") { $env:RUNPOD_API_KEY = $Matches[1] }
        }
    }
}
if (-not $env:RUNPOD_API_KEY) {
    Write-Error "RUNPOD_API_KEY not set. Add it to your environment or a .env file."
    exit 1
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

switch ($Command) {

    "build" {
        Write-Host "`n==> Building $FULL_IMAGE ..."
        docker build `
            --build-arg MODEL_ID_ARG=$MODEL_ID `
            -t $FULL_IMAGE `
            .
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
            Write-Host "Found endpoint $id — updating image to $FULL_IMAGE ..."
            runpodctl serverless update `
                --id $id `
                --image-name $FULL_IMAGE
        } else {
            Write-Host "No existing endpoint — creating '$ENDPOINT_NAME' ..."
            runpodctl serverless create `
                --name $ENDPOINT_NAME `
                --image-name $FULL_IMAGE `
                --gpu-ids "$GPU_IDS" `
                --workers-min $MIN_WORKERS `
                --workers-max $MAX_WORKERS `
                --env "MODEL_ID=$MODEL_ID" `
                --env "HF_HOME=/workspace/hf_cache"
        }

        Write-Host "`n==> Endpoint status:"
        runpodctl serverless list --output=table
    }

    "test" {
        Write-Host "`n==> Looking up endpoint ID for '$ENDPOINT_NAME' ..."
        $ep = runpodctl serverless list --output=json 2>$null | ConvertFrom-Json |
              Where-Object { $_.name -eq $ENDPOINT_NAME } |
              Select-Object -First 1

        if (-not $ep) { Write-Error "Endpoint '$ENDPOINT_NAME' not found. Run .\deploy.ps1 deploy first."; exit 1 }

        $endpointId = $ep.id
        $runUrl = "https://api.runpod.ai/v2/$endpointId/runsync"

        Write-Host "==> Sending test_input.json to $runUrl ..."
        $body = Get-Content test_input.json -Raw

        $response = Invoke-RestMethod `
            -Method Post `
            -Uri $runUrl `
            -Headers @{ "Authorization" = "Bearer $env:RUNPOD_API_KEY"; "Content-Type" = "application/json" } `
            -Body $body

        # Print everything except the (potentially huge) base64 mesh
        $preview = $response | ConvertTo-Json -Depth 5
        if ($preview.Length -gt 2000) {
            $preview = $preview.Substring(0, 2000) + "`n... (truncated)"
        }
        Write-Host "`n==> Response:`n$preview"

        # Save mesh to disk if present
        if ($response.output.mesh_b64) {
            $fmt = $response.output.format
            $outFile = "output_test.$fmt"
            [System.IO.File]::WriteAllBytes(
                (Join-Path $PWD $outFile),
                [System.Convert]::FromBase64String($response.output.mesh_b64)
            )
            Write-Host "`n==> Mesh saved to $outFile ($($response.output.face_count) faces)"
        }
    }

    "pod" {
        # Spin up an interactive on-demand pod for experimentation
        Write-Host "`n==> Creating on-demand pod with $FULL_IMAGE ..."
        runpodctl pod create `
            --name "${IMAGE_NAME}-dev" `
            --image-name $FULL_IMAGE `
            --gpu-type "$GPU_IDS" `
            --container-disk-size 50 `
            --ports "8888/http,22/tcp"

        Write-Host "`n==> Pod list:"
        runpodctl pod list --output=table
    }

    "status" {
        runpodctl serverless list --output=table
    }
}
