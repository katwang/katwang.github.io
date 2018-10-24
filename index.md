---
layout: default
title: Home
---

<header>
<h1>Hi, I'm Katherine!</h1>
<!-- template designed by <a href="http://html5up.net">HTML5 UP</a>.</h1> 
<p>I'm a current Master's student & aspiring data scientist.</p> -->

</header>

<section>
<p><span class="image left"><img src="images/pic15.jpg" alt="" /></span> Current Master's student, graduating December 2018.<br />
Aspiring data scientist | Python and R <br />
Two years of experience
<br />
Certified<br />
Thanks for your interest in my projects! You can find more at: 

<ul class="icons">
<li><a href="https://www.linkedin.com/in/{{ site.linkedin_username }}" class="icon style1 fa-linkedin"><span class="label">LinkedIn</span></a></li>
<li><a href="https://github.com/{{ site.github_username }}" class="icon style1 fa-github"><span class="label">GitHub</span></a></li>
<li><a href="mailto:{{ site.email }}" class="icon style1 fa-envelope-o"><span class="label">Email</span></a></li>
</ul>
</p>
</section>

<section>

<h2>My Projects</h2>

{% include tiles.html %}
</section>