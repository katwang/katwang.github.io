---
layout: page
title:  Interactive Map - US Opioid Prescribing Rate
---

**Dataset**: Medicare Part D Prescription Drug Program
<br/>
**Language**: R

An interactive <a href="https://github.com/katwang/Examples/blob/master/leafletmap.html">map</a> showing the opioid prescribing rate across the US in 2015.

The map was built using the <code>leaflet</code> package in R, with <a href="https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data">data</a> from the CDC. The data wrangling was performed using <code>sparklyr</code>, an R interface to Apache Spark.

<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width="600" height="600" src="https://raw.githubusercontent.com/katwang/Examples/master/leafletmap.html"></iframe>
