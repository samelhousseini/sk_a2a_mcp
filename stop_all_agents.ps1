# PowerShell script to stop all customer support agents, the orchestrator, and the MCP server
# Usage: 
#   .\stop_all_agents.ps1                  # Stop all agents
#   .\stop_all_agents.ps1 -AgentName FAQ   # Stop only the FAQ agent
#   .\stop_all_agents.ps1 -AgentName Orchestrator  # Stop only the Orchestrator agent
#   .\stop_all_agents.ps1 -AgentName MCPServer     # Stop only the MCP server
# 
# To restart agents, use start_all_agents.ps1 with -Restart:
#   .\start_all_agents.ps1 -Restart                # Restart all agents
#   .\start_all_agents.ps1 -AgentName FAQ -Restart # Restart only the FAQ agent

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Orchestrator", "FAQ", "TechTroubleshooting", "HumanEscalation", "MCPServer")]
    [string]$AgentName = ""
)

$AgentJobMapping = @{
    "Orchestrator" = "Python_Orchestrator_Agent"
    "FAQ" = "Python_FAQ_Agent"
    "TechTroubleshooting" = "Python_TechTroubleshooting_Agent"
    "HumanEscalation" = "Python_HumanEscalation_Agent"
    "MCPServer" = "Python_MCPServer_Agent"
}

$AgentJobNames = @()

if ([string]::IsNullOrEmpty($AgentName)) {
    # Stop all agents if no specific agent specified
    $AgentJobNames = @(
        "Python_Orchestrator_Agent",
        "Python_FAQ_Agent",
        "Python_TechTroubleshooting_Agent",
        "Python_HumanEscalation_Agent",
        "Python_MCPServer_Agent"
    )
    Write-Host "Attempting to stop ALL customer support agent jobs..."
} else {
    # Stop only the specified agent
    $AgentJobNames = @($AgentJobMapping[$AgentName])
    Write-Host "Attempting to stop only the $AgentName agent job ($($AgentJobMapping[$AgentName]))..."
}

foreach ($JobName in $AgentJobNames) {
    Write-Host "Looking for job: $JobName"
    $ServerJob = Get-Job -Name $JobName -ErrorAction SilentlyContinue

    if ($ServerJob) {
        Write-Host "Found job: $($ServerJob.Name) (State: $($ServerJob.State), ID: $($ServerJob.Id))"
        
        try {
            Stop-Job -Job $ServerJob -PassThru -ErrorAction Stop
            Write-Host "Stop command issued for job: $($ServerJob.Name)"
            
            Remove-Job -Job $ServerJob -ErrorAction Stop
            Write-Host "Job has been removed: $($ServerJob.Name)"
        } catch {
            Write-Error "Error stopping or removing job $($ServerJob.Name): $_"
        }
    } else {
        Write-Warning "No active job found with the name '$JobName'. It might not be running or was started differently."
    }
    Write-Host "---"
}

if ([string]::IsNullOrEmpty($AgentName)) {
    Write-Host "All agent jobs have been processed for stopping."
} else {
    Write-Host "The $AgentName agent job has been processed for stopping."
}

Write-Host "To start agents again, use: .\start_all_agents.ps1 [-AgentName <agent>]"
Write-Host "To restart agents, use: .\start_all_agents.ps1 [-AgentName <agent>] -Restart"
Start-Sleep -Seconds 3
