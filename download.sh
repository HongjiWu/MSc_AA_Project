
## WGET ##
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id=1VLIqKFwmXyy3_L-9ScmxLI-RGBxGrwdj' -O- \
			         | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O 'data.zip' \
			         'https://docs.google.com/uc?export=download&id=1VLIqKFwmXyy3_L-9ScmxLI-RGBxGrwdj&confirm='$(<confirm.txt)
