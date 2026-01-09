# setup_daily_task.ps1
# ==================================================
# 🔧 КОНФИГУРАЦИЯ — измените эти значения под свой проект
# ==================================================

# Корень проекта (должен содержать .venv и папку с задачей)
# Пример: "C:\projects\my-rag-app"
$ProjectRoot = "C:\projects\your-project-name"

# Папка с целевым скриптом (относительно ProjectRoot)
$TaskSubDir = "task6"

# Имя Python-скрипта для запуска
$ScriptFileName = "update_index.py"

# Время запуска (ежедневно в указанное время)
$RunTime = "02:00"  # формат: "ЧЧ:ММ" (24-часовой)

# Имя задачи в Планировщике
$TaskName = "RAG Daily Index Update"

# ==================================================
# 🚀 НИЖЕ НЕ ТРЕБУЕТСЯ ИЗМЕНЕНИЙ (автоматическая настройка)
# ==================================================

$TaskScriptDir = Join-Path $ProjectRoot $TaskSubDir
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TargetScript = Join-Path $TaskScriptDir $ScriptFileName

# Проверка существования файлов
if (-not (Test-Path $PythonExe)) {
    Write-Error "❌ Python из виртуального окружения не найден: $PythonExe"
    exit 1
}
if (-not (Test-Path $TargetScript)) {
    Write-Error "❌ Целевой скрипт не найден: $TargetScript"
    exit 1
}

# Парсинг времени
try {
    $Time = [DateTime]::ParseExact($RunTime, "HH:mm", $null)
} catch {
    Write-Error "❌ Неверный формат времени. Используйте 'HH:mm', например: '02:00'"
    exit 1
}

# Действие: запуск Python напрямую
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$TargetScript`"" -WorkingDirectory $TaskScriptDir

# Триггер: ежедневно в указанное время
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

# Настройки задачи
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -StartWhenAvailable

# Запуск от текущего пользователя (интерактивная сессия)
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Удаление старой задачи, если существует
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "🔄 Старая задача '$TaskName' удалена." -ForegroundColor Yellow
}

# Регистрация новой задачи
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Principal $Principal -Description "Ежедневное обновление векторного индекса RAG"

Write-Host "✅ Задача '$TaskName' создана!" -ForegroundColor Green
Write-Host "   • Скрипт: $TargetScript"
Write-Host "   • Время запуска: ежедневно в $RunTime"
Write-Host "   • Рабочая директория: $TaskScriptDir"