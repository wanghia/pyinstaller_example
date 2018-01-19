# pyinstaller_example

pyinstaller  打包当前的python代码，可提交到没有安装python的环境下运行

配置文件run.spec：
Analysis 第一个参数表示程序运行的入口
第二个参数表示路径
第三个参数表示需要一同打包的依赖包，该依赖包包含python3.6下auto-sklearn，sklearn等。若需要读取数据可将相对路径放在该参数目录下

打包运行pyinstaller run.spec

