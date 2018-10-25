---
layout: default
title: Home
---

<header>
<!--<h1>Hi, I'm Katherine!</h1>
 template designed by <a href="http://html5up.net">HTML5 UP</a>.</h1> 
<p>I'm a current Master's student & aspiring data scientist.</p> -->

</header>

<section>
<p><span class="image left"><img src="images/pic15.jpg" alt="" /></span> Hi there! I'm studying Health Data Science at the Harvard T.H. Chan School of Public Health, graduating December 2018. Previously worked at the FDA doing systems development and project management. Always excited about data and cute animal videos. </p>

<p>Thanks for your interest in my projects! You can find more at: <br />
	<a href="https://www.linkedin.com/in/{{ site.linkedin_username }}">LinkedIn</a> &nbsp;&nbsp; // &nbsp;&nbsp; <a href="https://github.com/{{ site.github_username }}">Github</a> &nbsp;&nbsp; // &nbsp;&nbsp; <a href="mailto:{{ site.email }}">E-Mail</a> 
</p>

</section>

<section>

<h2>My Projects</h2>

{% include tiles.html %}
</section>