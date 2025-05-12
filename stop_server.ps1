#Requires -Version 5.0

$JobName = "PythonSimpleServer"

Write-Host "Attempting to stop the server job: $JobName"

# Find the job
$ServerJob = Get-Job -Name $JobName -ErrorAction SilentlyContinue

if ($ServerJob) {
    Write-Host "Found job: $($ServerJob.Name) (State: $($ServerJob.State), ID: $($ServerJob.Id))"
    
    # Stop the job
    try {
        Stop-Job -Job $ServerJob -PassThru
        Write-Host "Stop command issued for job: $($ServerJob.Name)"
        
        # Remove the job from the session history
        # It's good practice to remove it after stopping, especially if it's completed or failed.
        Remove-Job -Job $ServerJob
        Write-Host "Job has been removed: $($ServerJob.Name)"
    } catch {
        Write-Error "Error stopping or removing job $($ServerJob.Name): $_"
    }
} else {
    Write-Warning "No active job found with the name '$JobName'. The server might not be running or was started differently."
}

# Add a small delay for user to read output before prompt returns if running interactively.
Start-Sleep -Seconds 3
