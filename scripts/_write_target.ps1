# PowerShell script to write d10s_mfe_trailing.py
 = "scripts\d10s_mfe_trailing.py"

 = @"
This is a test
"@

[System.IO.File]::WriteAllText(, .Replace([char]13+[char]10, [char]10), [System.Text.UTF8Encoding]::new())
Write-Host "Done"