$pat = 'chunk T='
Get-ChildItem 'D:\Projects\yttri-inference\logs\*.log' | Select-String -SimpleMatch -Pattern $pat |
  Select-Object -Last 40 | ForEach-Object { "$($_.Filename): $($_.Line)" }
