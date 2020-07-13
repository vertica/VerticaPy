<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>Why VerticaPy ?</h1>
        <p>
          Nowadays, 'Big Data' is one of the main topics in the data science world, and data scientists are often at the center of any organization. The benefits of becoming more data-driven are undeniable and are often needed to survive in the industry.
        </p><br>
        <p>
          Vertica was the first real analytic columnar database and is still the fastest in the market. However, SQL alone isn't flexible enough to meet the needs of data scientists. Python has quickly become the most popular tool in this domain, owing much of its flexibility to its high-level of abstraction and impressively large and ever-growing set of libraries. Its accessibility has led to the development of popular and perfomant APIs, like pandas and scikit-learn, and a dedicated community of data scientists.
        </p><br>
          <p>However, Python only works in-memory for a single node process. While distributed programming languages have tried to face this challenge, they are still generally in-memory and can never hope to process all of your data, and moving data is expensive. On top of all of this, data scientists must also find convenient ways to deploy their data and models. The whole process is time consuming.</p><br>
        <p>
          VerticaPy aims to solve all of these problems. The idea is simple: instead of moving data to your tools, VerticaPy brings your tools to the data.</b>
        </p><br>
        <center>
          <img src="img/VerticaPy.png" width="80%">
        </center>
        <h1>History</h1>
        <p>
            When the first data science technologies and tools came onto the scene, optimization wasn't a high priority. Companies didn't pay much mind to how the needs of data storage and ingestion might change. Back then, databases were still used as data warehouses, and moving data around was often impossible without making compromises in security.
        </p><br>
        <p>
          To address these problems, Vertica implemented the first in-database, scalable machine learning algorithms. That was back in 2015, and other databases have been trying to catch up ever since.
        </p><br>
        <p>
          However, what SQL has in scalability, it lacks in flexibility. Python has the opposite problem: it's highly flexible, but not scalable. The idea of combining the strengths of these technologies came about in 2017 by Vertica data scientist Badr Quali and, after 3 years of development, became unique and powerful library, VerticaPy.
    </p>
        <h1>First Official Logo</h1>
        <p>The first of the VerticaPy logo:</p><br>
        <center>
          <img src="img/first-logo.png" width="30%" style="min-width: 300px;">
        </center>
        <h1>A Few Words from the Creator</h1>
        <p>"This Python Module is the result of my passion for Data Science. I love discovering everything possible in the data. I always kept a passion for mathematics and specially for statistics. When I saw the lack of libraries using as back-end the power of columnar MPP Database, I decided to help the Data Science Community by bringing the logic to the data."</p><br>
        <center>
          <img src="img/Badr.jpg" width="30%" style="min-width: 300px;">
        </center>
        <div><?php include('include/footer.php'); ?></div>
      </div>
    </body>
</html>
