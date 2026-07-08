# launch_all_runs.ps1
# Launches all 12 cross-task generalization experiment runs in parallel.
# Each run is a separate background job with its own log file.

$ErrorActionPreference = "Continue"
$ProjectRoot = "c:\Users\charu\OneDrive\Desktop\transcription_project"
$LogDir = "$ProjectRoot\experiments\cross_task_generalization\runs\launch_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Define all 12 runs
$runs = @(
    # Task 2 CHILDES - gpt-5.4-mini
    @{ Task="task2_childes"; Condition="c0"; Model="gpt-5.4-mini" },
    @{ Task="task2_childes"; Condition="c1"; Model="gpt-5.4-mini" },
    @{ Task="task2_childes"; Condition="c2"; Model="gpt-5.4-mini" },
    # Task 2 CHILDES - gpt-5-mini
    @{ Task="task2_childes"; Condition="c0"; Model="gpt-5-mini" },
    @{ Task="task2_childes"; Condition="c1"; Model="gpt-5-mini" },
    @{ Task="task2_childes"; Condition="c2"; Model="gpt-5-mini" },
    # Task 3 TIMIT - gpt-5.4-mini
    @{ Task="task3_timit"; Condition="c0"; Model="gpt-5.4-mini" },
    @{ Task="task3_timit"; Condition="c1"; Model="gpt-5.4-mini" },
    @{ Task="task3_timit"; Condition="c2"; Model="gpt-5.4-mini" },
    # Task 3 TIMIT - gpt-5-mini
    @{ Task="task3_timit"; Condition="c0"; Model="gpt-5-mini" },
    @{ Task="task3_timit"; Condition="c1"; Model="gpt-5-mini" },
    @{ Task="task3_timit"; Condition="c2"; Model="gpt-5-mini" }
)

$jobs = @()

foreach ($run in $runs) {
    $task = $run.Task
    $condition = $run.Condition
    $model = $run.Model
    $logFile = "$LogDir\${task}_${condition}_${model}_${timestamp}.log"
    $scriptPath = "$ProjectRoot\experiments\cross_task_generalization\$task\run.py"

    Write-Host "Launching: $task $condition $model -> $logFile"

    $job = Start-Job -ScriptBlock {
        param($python, $script, $condition, $model, $root)
        Set-Location $root
        & $python $script --condition $condition --model $model 2>&1
    } -ArgumentList "python", $scriptPath, $condition, $model, $ProjectRoot

    $jobs += @{
        Job = $job
        Task = $task
        Condition = $condition
        Model = $model
        LogFile = $logFile
    }
}

Write-Host "`nAll $($jobs.Count) runs launched. Waiting for completion..."
Write-Host "Logs will be saved to: $LogDir`n"

# Wait for all jobs and save logs
$completed = 0
while ($completed -lt $jobs.Count) {
    foreach ($entry in $jobs) {
        $job = $entry.Job
        if ($job.State -eq "Completed" -or $job.State -eq "Failed" -or $job.State -eq "Stopped") {
            if (-not $entry.ContainsKey("Done")) {
                $output = Receive-Job -Job $job
                $output | Out-File -FilePath $entry.LogFile -Encoding utf8
                $status = if ($job.State -eq "Completed") { "SUCCESS" } else { "FAILED ($($job.State))" }
                Write-Host "[$status] $($entry.Task) $($entry.Condition) $($entry.Model)"
                $entry["Done"] = $true
                $completed++
            }
        }
    }
    if ($completed -lt $jobs.Count) {
        Start-Sleep -Seconds 10
    }
}

# Summary
Write-Host "`n=== ALL RUNS COMPLETE ==="
foreach ($entry in $jobs) {
    $status = if ($entry.Job.State -eq "Completed") { "OK" } else { $entry.Job.State }
    Write-Host "  $($entry.Task) $($entry.Condition) $($entry.Model): $status"
}

# Cleanup jobs
$jobs | ForEach-Object { Remove-Job -Job $_.Job -Force }

Write-Host "`nDone. Check logs in: $LogDir"
