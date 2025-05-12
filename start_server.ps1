#Requires -Version 5.0
# Define paths relative to the script's location
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvPath = Join-Path $PSScriptRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe" # Use ".exe" for Windows
$ServerScript = Join-Path $PSScriptRoot "a2a_agents\simple_server.py"
$OutputFile = Join-Path $PSScriptRoot "simple_server.out"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

# Check if Python executable exists
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe. Please ensure the virtual environment is set up correctly."
    exit 1
}

# Check if server script exists
if (-not (Test-Path $ServerScript)) {
    Write-Error "Server script not found at $ServerScript. Please check the path."
    exit 1
}

# Script block to run as a background job
$ScriptBlock = {
    param($PythonExeParam, $ServerScriptParam, $OutputFileParam, $ActivateScriptParamPSScriptRoot)

    # PowerShell's execution policy might prevent running Activate.ps1 if it's not signed
    # and the policy is AllSigned or Restricted.
    # For running a Python script from a venv, directly calling the venv's python.exe is often sufficient
    # and avoids issues with execution policy for the activation script itself within the job.

    # Ensure the output directory exists (though for root, it will)
    $OutputDirectory = Split-Path -Parent $OutputFileParam
    if (-not (Test-Path $OutputDirectory)) {
        New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
    }

    Write-Output "Job: Starting server $PythonExeParam $ServerScriptParam"
    Write-Output "Job: Output will be redirected to $OutputFileParam"

    try {
        # Using Start-Process to detach more thoroughly, but Start-Job is usually sufficient
        # and integrates with PS job management.
        # For Start-Job, the '&' call operator is used.
        # The *> operator redirects all output streams.
        & $PythonExeParam $ServerScriptParam *> $OutputFileParam
        Write-Output "Job: Server process launched."
    } catch {
        Write-Error "Job: Error starting server: $_"
        # Attempt to log the error to the output file as well
        $ErrorMessage = "Job Error: $($_.Exception.Message)"
        Add-Content -Path $OutputFileParam -Value $ErrorMessage
    }
}

# Remove a job with the same name if it already exists and is in a terminal state (Completed, Failed, Stopped)
$ExistingJob = Get-Job -Name "PythonSimpleServer" | Where-Object { $_.State -in @('Completed', 'Failed', 'Stopped') }
if ($ExistingJob) {
    Write-Warning "Removing previous terminal job: $($ExistingJob.Name) (State: $($ExistingJob.State))"
    Remove-Job -Job $ExistingJob
}

# Start the background job
Write-Host "Attempting to start Python server as a background job..."
try {
    Start-Job -ScriptBlock $ScriptBlock -ArgumentList $PythonExe, $ServerScript, $OutputFile, $ActivateScript, $PSScriptRoot -Name "PythonSimpleServer"
    Write-Host "Server start command issued as a background job named 'PythonSimpleServer'."
    Write-Host "Output should be redirected to: $OutputFile"
    Write-Host ""
    Write-Host "To check job status, run: Get-Job -Name PythonSimpleServer"
    Write-Host "To see live output (if any is being actively written and not just to file), or errors from the job itself:"
    Write-Host "  Receive-Job -Name PythonSimpleServer -Keep"
    Write-Host "To stop the server (stops the job), run: Stop-Job -Name PythonSimpleServer; Remove-Job -Name PythonSimpleServer"
    Write-Host "The server's own logs will be in '$OutputFile'"
} catch {
    Write-Error "Failed to start job: $_"
}

# The .ps1 script will now exit, but the job should continue running.
# Add a small delay for user to read output before prompt returns if running interactively.
Start-Sleep -Seconds 3
