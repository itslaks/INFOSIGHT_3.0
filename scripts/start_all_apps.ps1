# Start all INFOSIGHT 3.0 app modules in standalone mode (Windows PowerShell)
# Usage: .\scripts\start_all_apps.ps1

$projectRoot = Resolve-Path -Path ".."  # run from scripts folder
Set-Location $projectRoot

$appMap = @{
    infocrypt      = 5001
    cybersentry_ai = 5002
    donna          = 5003
    enscan         = 5004
    filescanner    = 5005
    infosight_ai   = 5006
    inkwell_ai     = 5007
    nova_ai        = 5008
    osint          = 5009
    portscanner    = 5010
    snapspeak_ai   = 5011
    trueshot_ai    = 5012
    webseeker      = 5013
}

if (-not (Test-Path -Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }

foreach ($key in $appMap.Keys) {
    $port = $appMap[$key]
    $env = @{
        APP_HOST = '127.0.0.1'
        APP_PORT = [string]$port
    }
    $script = Join-Path $projectRoot "app\$key.py"
    $stdout = Join-Path $projectRoot "logs\$key.log"
    Write-Host "Starting $key on 127.0.0.1:$port"
    Start-Process -FilePath (Get-Command python).Source -ArgumentList $script -WorkingDirectory $projectRoot -Environment $env -RedirectStandardOutput $stdout -RedirectStandardError $stdout
}

# Start main gateway server on 5000 by default
$gatewayHost = $env:SERVER_HOST -or '127.0.0.1'
$gatewayPort = $env:SERVER_PORT -or '5000'
Write-Host "Starting main server gateway on $gatewayHost:$gatewayPort"
Start-Process -FilePath (Get-Command python).Source -ArgumentList "server.py --mode distributed --host $gatewayHost --port $gatewayPort" -WorkingDirectory $projectRoot

Write-Host "All processes launched. Use Task Manager to inspect or Stop-Process to kill."