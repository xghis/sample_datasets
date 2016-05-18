logLevel := Level.Warn
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.13.0")

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

resolvers += "sonatype-releases" at "https://oss.sonatype.org/content/repositories/releases/"

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.3")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "0.8.0")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.1.0")
