---
layout: page
title: 
---

<ul class="posts">
  {% for post in site.posts %}
      {% unless post.img != "essay" %}
        <div style="margin-top:0%;">
        {% unless post.next %}
          <h2>{{ post.date | date: '%Y' }}</h2>
        {% else %}
          {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
          {% capture nyear %}{{ post.next.date | date: '%Y' }}{% endcapture %}
          {% if year != nyear %}
            <h2>{{ post.date | date: '%Y' }}</h2>
          {% endif %}
        {% endunless %}
      </div>
        <li itemscope>
          <div style="margin-top:0%;">
            <a href="{{ site.github.url }}{{ post.url }}" style="text-decoration:none;">{{ post.title }}</a>
            <span class="post-date"> {{ post.date | date: "%B %-d" }}</span>
            <!-- {% if post.img == "/assets/chitriangles_vecs.png" %}
              <img src="{{ post.img }}" align="right" width="100">
            {% elsif post.img == "/assets/clock-regular.svg" %}
              <img src="{{ post.img }}" align="right" width="80">
            {% elsif post.img == "/assets/gplvm_parabola_results.png" %}
              <img src="{{ post.img }}" align="right" width="200">
            {% elsif post.img != "" %}
              <img src="{{ post.img }}" align="right" width="150">
            {% else %}
              <img src="/assets/dice-six-solid.svg" align="right" width="40" style="opacity: 0.5;">
            {% endif %} -->
            <p class="post-date">{{ post.blurb }}</p>
          </div>
        </li>
      {% endunless %}
  {% endfor %}
</ul>
