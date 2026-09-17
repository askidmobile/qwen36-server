param([int]$Words = 18500, [string]$Out = 'D:\Projects\yttri-inference\prompt_long128k.txt')
$sb = New-Object System.Text.StringBuilder
[void]$sb.Append('Session 128k probe. ')
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
[void]$sb.Append('Summarize the pattern in one sentence.')
[System.IO.File]::WriteAllText($Out, $sb.ToString(), [System.Text.UTF8Encoding]::new($false))
$fi = Get-Item $Out
Write-Output ("WROTE {0} bytes={1}" -f $fi.FullName, $fi.Length)
