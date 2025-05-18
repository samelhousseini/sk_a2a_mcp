# PowerShell script to start all customer support agents, the orchestrator, and the MCP server
# Usage: 
#   .\start_all_agents.ps1                         # Start all agents
#   .\start_all_agents.ps1 -AgentName FAQ          # Start only the FAQ agent
#   .\start_all_agents.ps1 -AgentName Orchestrator # Start only the Orchestrator agent
#   .\start_all_agents.ps1 -AgentName MCPServer    # Start only the MCP server
#   .\start_all_agents.ps1 -Restart                # Restart all agents 
#   .\start_all_agents.ps1 -AgentName FAQ -Restart # Restart only the FAQ agent

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Orchestrator", "FAQ", "TechTroubleshooting", "HumanEscalation", "MCPServer")]
    [string]$AgentName = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$Restart = $false
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PSScriptRoot = Join-Path $ProjectRoot "app" # Assuming this script is in the 'app' directory
Write-Host "PSScriptRoot is: $PSScriptRoot" -ForegroundColor Cyan

$VenvPath = Join-Path $PSScriptRoot "..\.venv" # Assuming .venv is in the parent directory (project root)
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

# Agent Scripts - ensure these paths are correct relative to this script's location (app folder)
$OrchestratorScript = Join-Path $PSScriptRoot "orchestrator_agent.py"
$FAQScript = Join-Path $PSScriptRoot "faq_agent.py"
$TechScript = Join-Path $PSScriptRoot "technical_troubleshooting_agent.py"
$EscalationScript = Join-Path $PSScriptRoot "human_support_escalation_agent.py"
$MCPServerScript = Join-Path $PSScriptRoot "mcp_tools_server.py"

# Output directory and files for logs
$OutputDir = Join-Path $PSScriptRoot "cli-output"
Write-Host "OutputDir is: $OutputDir" -ForegroundColor Cyan

# Create the directory if it doesn't exist
if (-not (Test-Path -Path $OutputDir -PathType Container)) {
    try {
        New-Item -Path $OutputDir -ItemType Directory -Force -ErrorAction Stop | Out-Null
        Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
    }
    catch {
        Write-Host "Error creating directory: $_" -ForegroundColor Red
    }
}
else {
    Write-Host "Directory already exists: $OutputDir" -ForegroundColor Yellow
}

# Output files for logs
$OrchestratorOut = Join-Path $OutputDir "orchestrator_agent.out"
$FAQOut = Join-Path $OutputDir "faq_agent.out"
$TechOut = Join-Path $OutputDir "technical_troubleshooting_agent.out"
$EscalationOut = Join-Path $OutputDir "human_support_escalation_agent.out"
$MCPServerOut = Join-Path $OutputDir "mcp_tools_server.out"

# Ports (ensure these match agent configurations and orchestrator client URLs)
$OrchestratorPort = 8000
$FAQPort = 8001
$TechPort = 8002
$EscalationPort = 8003
$MCPServerPort = 8010

# Function to start an agent
function Start-Agent {
    param (
        [string]$AgentName,
        [string]$ScriptPath,
        [string]$OutputFile,
        [int]$Port
    )

    if (-not (Test-Path $ScriptPath)) {
        Write-Error "$AgentName script not found at $ScriptPath. Please check the path."
        return
    }

    $JobName = "Python_${AgentName}_Agent"

    # Remove a job with the same name if it already exists and is in a terminal state
    $ExistingJob = Get-Job -Name $JobName | Where-Object { $_.State -in @('Completed', 'Failed', 'Stopped') }
    if ($ExistingJob) {
        Write-Warning "Removing previous terminal job: $($ExistingJob.Name) (State: $($ExistingJob.State))"
        Remove-Job -Job $ExistingJob
    }

    $ScriptBlock = {
        param($PythonExeParam, $ScriptPathParam, $OutputFileParam, $PortParam, $AgentNameParam)
        
        Write-Output "Job ($AgentNameParam): Starting server $PythonExeParam $ScriptPathParam --port $PortParam"
        Write-Output "Job ($AgentNameParam): Output will be redirected to $OutputFileParam"
        
        # In a real scenario, the Python scripts would need to accept --port argument
        # For now, we assume they are hardcoded or configured to use these ports.
        # The A2AServer in the examples can take host/port arguments in its constructor.
        # The __main__ blocks in the agent files would need to be updated to parse args and pass to server.
        # For simplicity, this script just calls the python script. The python script itself needs to handle the port.
        try {
            # Modify the agent's __main__ to accept port or pass it via environment variable if needed.
            # This example assumes the python script itself will pick up the correct port or is already configured.
            & $PythonExeParam $ScriptPathParam *> $OutputFileParam 
            Write-Output "Job ($AgentNameParam): Server process launched."
        } catch {
            Write-Error "Job ($AgentNameParam): Error starting server: $_"
            $ErrorMessage = "Job Error ($AgentNameParam): $($_.Exception.Message)"
            Add-Content -Path $OutputFileParam -Value $ErrorMessage
        }
    }

    Write-Host "Attempting to start $AgentName Agent as a background job..."
    try {
        Start-Job -ScriptBlock $ScriptBlock -ArgumentList $PythonExe, $ScriptPath, $OutputFile, $Port, $AgentName -Name $JobName
        Write-Host "$AgentName Agent start command issued as job '$JobName'. Output: $OutputFile"
    } catch {
        Write-Error "Failed to start $AgentName Agent job: $_"
    }
}

# Function to stop an agent
function Stop-Agent {
    param (
        [string]$AgentName
    )
    
    $JobName = "Python_${AgentName}_Agent"
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
}

# Check Python Executable
if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found at $PythonExe. Ensure .venv is in project root and activated/used."
    exit 1
}

# If restart is specified, stop the agent(s) first
if ($Restart) {
    if ([string]::IsNullOrEmpty($AgentName)) {
        Write-Host "Stopping ALL agents before restarting..."
        Stop-Agent -AgentName "Orchestrator"
        Stop-Agent -AgentName "FAQ"
        Stop-Agent -AgentName "TechTroubleshooting"
        Stop-Agent -AgentName "HumanEscalation"
        Stop-Agent -AgentName "MCPServer"
    } else {
        Write-Host "Stopping $AgentName agent before restarting..."
        Stop-Agent -AgentName $AgentName
    }
    # Wait a moment for processes to fully terminate
    Start-Sleep -Seconds 3
}

if ([string]::IsNullOrEmpty($AgentName)) {
    Write-Host "Starting ALL Customer Support Agents..."

    # Start MCP Server First
    Start-Agent -AgentName "MCPServer" -ScriptPath $MCPServerScript -OutputFile $MCPServerOut -Port $MCPServerPort
    Start-Sleep -Seconds 2 # Give a moment for the server to start

    # Start Helper Agents
    Start-Agent -AgentName "FAQ" -ScriptPath $FAQScript -OutputFile $FAQOut -Port $FAQPort
    Start-Sleep -Seconds 2 # Give a moment for the agent to start

    Start-Agent -AgentName "TechTroubleshooting" -ScriptPath $TechScript -OutputFile $TechOut -Port $TechPort
    Start-Sleep -Seconds 2

    Start-Agent -AgentName "HumanEscalation" -ScriptPath $EscalationScript -OutputFile $EscalationOut -Port $EscalationPort
    Start-Sleep -Seconds 2

    # Start Orchestrator Agent Last
    Start-Agent -AgentName "Orchestrator" -ScriptPath $OrchestratorScript -OutputFile $OrchestratorOut -Port $OrchestratorPort
} else {
    Write-Host "Starting only the $AgentName agent..."
    
    switch ($AgentName) {
        "Orchestrator" {
            Start-Agent -AgentName "Orchestrator" -ScriptPath $OrchestratorScript -OutputFile $OrchestratorOut -Port $OrchestratorPort
        }
        "FAQ" {
            Start-Agent -AgentName "FAQ" -ScriptPath $FAQScript -OutputFile $FAQOut -Port $FAQPort
        }
        "TechTroubleshooting" {
            Start-Agent -AgentName "TechTroubleshooting" -ScriptPath $TechScript -OutputFile $TechOut -Port $TechPort
        }
        "HumanEscalation" {
            Start-Agent -AgentName "HumanEscalation" -ScriptPath $EscalationScript -OutputFile $EscalationOut -Port $EscalationPort
        }
        "MCPServer" {
            Start-Agent -AgentName "MCPServer" -ScriptPath $MCPServerScript -OutputFile $MCPServerOut -Port $MCPServerPort
        }
    }
}

Write-Host ""
$action = if ($Restart) { "restart" } else { "start" }

if ([string]::IsNullOrEmpty($AgentName)) {
    Write-Host "All agent $($action) commands issued."
} else {
    Write-Host "The $AgentName agent $($action) command has been issued."
}
Write-Host "To check job statuses, run: Get-Job"
Write-Host "To see live output/errors from a job (e.g., FAQ Agent): Receive-Job -Name Python_FAQ_Agent -Keep"
Write-Host "To stop agents, run: .\stop_all_agents.ps1 [-AgentName <agent>]"
Write-Host "To restart agents, run: .\start_all_agents.ps1 [-AgentName <agent>] -Restart"
Write-Host "  or manually: Stop-Job -Name Python_*_Agent; Remove-Job -Name Python_*_Agent; .\start_all_agents.ps1"
Write-Host "Individual agent logs are in their respective .out files in the 'app' directory."

Start-Sleep -Seconds 5
