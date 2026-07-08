# launch_claude_runs.ps1
# Launches all 6 cross-task generalization experiment runs for Claude Haiku in parallel.
# Uses claude-haiku-4-20250514 (fastest & smallest Claude model).

$ErrorActionPreference = "Continue"
$ProjectRoot = "c:\Users\charu\OneDrive\Desktop\transcription_project"
$LogDir = "$ProjectRoot\experiments\cross_task_generalization\runs\launch_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Define all 6 Claude Haiku runs (3 conditions x 2 tasks)
$runs = @(
    # Task 2 CHILDES - claude-haiku-4-5-20251001
    @{ Task="task2_childes"; Condition="c0"; Model="claude-haiku-4-5-20251001" },
    @{ Task="task2_childes"; Condition="c1"; Model="claude-haiku-4-5-20251001" },
    @{ Task="task2_childes"; Condition="c2"; Model="claude-haiku-4-5-20251001" },
    # Task 3 TIMIT - claude-haiku-4-5-20251001
    @{ Task="task3_timit"; Condition="c0"; Model="claude-haiku-4-5-20251001" },
    @{ Task="task3_timit"; Condition="c1"; Model="claude-haiku-4-5-20251001" },
    @{ Task="task3_timit"; Condition="c2"; Model="claude-haiku-4-5-20251001" }
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
        if ($script -like "*task3_timit*" -and $condition -eq "c2") {
            & $python $script --condition $condition --model $model --parallel-records 5 --chunk-size 1 --parallel-chunks 1 --max-output-tokens 20000 2>&1
        } elseif ($script -like "*task3_timit*" -and $condition -in @("c0", "c1")) {
            & $python $script --condition $condition --model $model --parallel-records 5 --max-output-tokens 20000 2>&1
        } else {
            & $python $script --condition $condition --model $model 2>&1
        }
    } -ArgumentList "python", $scriptPath, $condition, $model, $ProjectRoot

    $jobs += @{
        Job = $job
        Task = $task
        Condition = $condition
        Model = $model
        LogFile = $logFile
    }
}

Write-Host "`nAll $($jobs.Count) Claude Haiku runs launched. Waiting for completion..."
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
Write-Host "`n=== ALL CLAUDE HAIKU RUNS COMPLETE ==="
foreach ($entry in $jobs) {
    $status = if ($entry.Job.State -eq "Completed") { "OK" } else { $entry.Job.State }
    Write-Host "  $($entry.Task) $($entry.Condition) $($entry.Model): $status"
}

# Cleanup jobs
$jobs | ForEach-Object { Remove-Job -Job $_.Job -Force }

Write-Host "`nDone. Check logs in: $LogDir"
