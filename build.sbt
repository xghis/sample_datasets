name := "cosinesimilarity"

version := "0.1"

scalaVersion := "2.10.5"
crossScalaVersions := Seq("2.10.5", "2.11.7")
sparkVersion := "1.6.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0" % "provided"
  "com.github.scopt" %% "scopt" % "3.4.0"
)
