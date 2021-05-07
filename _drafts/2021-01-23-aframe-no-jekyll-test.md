<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>A-Frame demo</title>
    <meta name="description" content="A-Frame demo">
    <script src="https://aframe.io/releases/1.1.0/aframe.min.js"></script>
    <script src="https://unpkg.com/aframe-environment-component/dist/aframe-environment-component.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/n5ro/aframe-physics-system@v4.0.1/dist/aframe-physics-system.min.js"></script>
  </head>
  <body>
    {% gtm body %}
    <a-scene physics="debug: true; gravity: 9.8">

        <a-assets>
            <!-- ... -->
            <img id="skyTexture" src="https://cdn.aframe.io/360-image-gallery-boilerplate/img/sechelt.jpg">
            <img id="groundTexture" src="https://cdn.aframe.io/a-painter/images/floor.jpg">
        </a-assets>

        <!-- Ground -->
        <a-plane static-body id="floor"></a-plane>

        <a-box id="#box1" color="red" position="-3 2 -5" rotation="0 45 45" scale="1 1 1"></a-box>
        <a-torus-knot id="#torKnot1" color="#B84A39" arc="90" p="2" q="5" radius="3" radius-tubular="0.2" position="-10 5 0"></a-torus-knot>
        <a-box dynamic-body="mass: 12" constraint="target: #floor; collideConnected: true"
            animation="property: object3D.position.y; to: 0.0; dir: alternate; dur: 2000; loop: true"
            id="#box2" src="https://i.imgur.com/mYmmbrp.jpg" position="5 10 -5" rotation="0 45 45"
            scale="2 2 2"></a-box>

        <a-entity environment="preset: forest; dressingAmount: 200"></a-entity> <!-- skyType=gradient; skyColor=#FFAABB00 -->

        <a-light type="ambient" color="#445451"></a-light>
        <a-light type="point" intensity="2" position="2 4 4"></a-light>

        <a-camera>
            <a-cursor></a-cursor>
        </a-camera>

      </a-scene>

      <script>
        var boxEl = document.getElementById('#box1');
        boxEl.addEventListener('mouseenter', function () {
          boxEl.setAttribute('scale', {x: Math.random() * 3, y: Math.random() * 3, z: Math.random() * 3});
        });

        var torKnotEl = document.getElementById('#torKnot1');
        torKnotEl.addEventListener('mouseenter', function () {
            const { z } = torKnotEl.getAttribute("rotation");
            // console.log(z);
            torKnotEl.setAttribute('rotation', {x: 0, y: 45, z: z + 45});
        });
      </script>
  </body>
</html>
