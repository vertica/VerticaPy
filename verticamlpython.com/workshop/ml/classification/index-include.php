<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Classification">Classification<a class="anchor-link" href="#Classification">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Classifications are specific ML algorithms which are used to predict categorical response columns. When the number of categories of the response column is greater than two, we use the term 'Multiclass Classification' to describe them. Examples of classification are predicting the flower species using specific characteristics or predicting telco customers churn.</p>
<p>To understand how to create a classification model, let's predict the flower species using the Iris dataset.</p>
<p>First, let's import the Random Forest Classifier.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can create a model object. As Vertica has its own model management system, we need to choose a model name with the other parameters.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="s2">&quot;RF_Iris&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can fit the model using the corresponding data.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[55]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;SepalLengthCm&quot;</span><span class="p">],</span> <span class="s2">&quot;Species&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[55]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>

===========
call_string
===========
SELECT rf_classifier(&#39;public.RF_Iris&#39;, &#39;iris&#39;, &#39;&#34;species&#34;&#39;, &#39;&#34;PetalLengthCm&#34;, &#34;SepalLengthCm&#34;&#39; USING PARAMETERS exclude_columns=&#39;&#39;, ntree=10, mtry=1, sampling_size=0.632, max_depth=5, max_breadth=1000000000, min_leaf_size=1, min_info_gain=0, nbins=32);

=======
details
=======
  predictor  |      type      
-------------+----------------
petallengthcm|float or numeric
sepallengthcm|float or numeric


===============
Additional Info
===============
       Name       |Value
------------------+-----
    tree_count    | 10  
rejected_row_count|  0  
accepted_row_count| 150 </pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To evaluate the model, we can use different metrics.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span><span class="o">.</span><span class="n">classification_report</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-setosa</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-versicolor</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>Iris-virginica</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9956000000000004</td><td style="border: 1px solid white;">0.9968000000000006</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>prc_auc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.992143797360809</td><td style="border: 1px solid white;">0.9936004843289119</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>accuracy</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9666666666666667</td><td style="border: 1px solid white;">0.9733333333333334</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>log_loss</b></td><td style="border: 1px solid white;">0.0126536049390048</td><td style="border: 1px solid white;">0.0404264391589519</td><td style="border: 1px solid white;">0.0352071123419837</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>precision</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.96</td><td style="border: 1px solid white;">0.98</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>recall</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9411764705882353</td><td style="border: 1px solid white;">0.9423076923076923</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>f1_score</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9600989792762139</td><td style="border: 1px solid white;">0.9654682104407882</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>mcc</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9254762227411247</td><td style="border: 1px solid white;">0.9410092614535137</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>informedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9209744503862152</td><td style="border: 1px solid white;">0.9321036106750391</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>markedness</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9299999999999999</td><td style="border: 1px solid white;">0.95</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>csi</b></td><td style="border: 1px solid white;">1.0</td><td style="border: 1px solid white;">0.9056603773584906</td><td style="border: 1px solid white;">0.9245283018867925</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>cutoff</b></td><td style="border: 1px solid white;">0.635</td><td style="border: 1px solid white;">0.526</td><td style="border: 1px solid white;">0.325</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[56]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We did not split the data into train and test which will be more relevant. The purpose is to understand all the possible metrics to evaluate a Classification. The most famous one is the Accuracy for which the closer it is to 1, the better it is. However, wrong metrics can lead to wrong interpretations.</p>
<p>Let's take the example where the purpose is to find Bank Frauds. As frauds are rare they can represent less than 1% of the data. Predicting that all the data do not correspond to Frauds will then lead to more than 99% of accuracy. That's why ROC AUC and PRC AUC are more robust metrics.</p>
<p>Besides, a good model is a model which will solve the Business problem. Most of the time we consider that any model better than the random model is good.</p>
<p>In the next lesson, you'll learn how to build unsupervised models.</p>

</div>
</div>
</div>