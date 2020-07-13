<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="LinearSVC">LinearSVC<a class="anchor-link" href="#LinearSVC">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">LinearSVC</span><span class="p">(</span><span class="n">name</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
          <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
          <span class="n">tol</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span> 
          <span class="n">C</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span> 
          <span class="n">fit_intercept</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
          <span class="n">intercept_scaling</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span>
          <span class="n">intercept_mode</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;regularized&quot;</span><span class="p">,</span>
          <span class="n">class_weight</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>
          <span class="n">max_iter</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">100</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Creates a LinearSVC object by using the Vertica Highly Distributed and Scalable SVM on the data. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Name of the model to be stored in the database.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">tol</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Used to control accuracy.</td> </tr>
    <tr> <td><div class="param_name">C</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The weight for misclassification cost. The algorithm minimizes the regularization cost and the misclassification cost.</td> </tr>
    <tr> <td><div class="param_name">fit_intercept</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>A bool to fit also the intercept.</td> </tr>
    <tr> <td><div class="param_name">intercept_scaling</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>A float value, serves as the value of a dummy feature whose coefficient Vertica uses to calculate the model intercept. Because the dummy feature is not in the training data, its values are set to a constant, by default set to 1. </td> </tr>
    <tr> <td><div class="param_name">intercept_mode</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Specify how to treat the intercept.<br><ul><li><b>regularized :</b> Fits the intercept and applies a regularization on it.</li><li><b>unregularized :</b> Fits the intercept but does not include it in regularization.</li></ul></td></tr>
    <tr> <td><div class="param_name">class_weight</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>Specifies how to determine weights of the two classes. It can be a list of 2 elements or one of the following method: <br> <ul><li><b>auto :</b> Weights each class according to the number of samples.</li><li><b>none :</b>  No weights are used.</li></ul></td> </tr>
    <tr> <td><div class="param_name">max_iter</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The maximum number of iterations that the algorithm performs.</td> </tr>
</table><h3 id="Attributes">Attributes<a class="anchor-link" href="#Attributes">&#182;</a></h3><p>After the object creation, all the parameters become attributes. The model will also create extra attributes when fitting the model:</p>
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th>  <th>Description</th> </tr>
    <tr> <td><div class="param_name">coef</div></td> <td><div class="type">tablesample</div></td> <td>Coefficients and their mathematical information (pvalue, std, value...)</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td>Train relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td>List of the predictors.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td>Response column.</td> </tr>
    <tr> <td><div class="param_name">test_relation</div></td> <td><div class="type">float</div></td> <td>Relation used to test the model. All the model methods are abstractions which will simplify the process. The test relation will be used by many methods to evaluate the model. If empty, the training relation will be used as test. You can change it anytime by changing the test_relation attribute of the object.</td> </tr>
</table><h3 id="Methods">Methods<a class="anchor-link" href="#Methods">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Description</th> </tr>
    <tr> <td><a href="../Classification/classification_report/index.php">classification_report</a></td> <td>Computes a classification report using multiple metrics to evaluate the model (AUC, accuracy, PRC AUC, F1...). In case of multiclass classification, it will consider each category as positive and switch to the next one during the computation.</td> </tr>
    <tr> <td><a href="../Classification/confusion_matrix/index.php">confusion_matrix</a></td> <td>Computes the model confusion matrix.</td> </tr>
    <tr> <td><a href="../Classification/deploySQL/index.php">deploySQL</a></td> <td>Returns the SQL code needed to deploy the model.</td> </tr>
    <tr> <td><a href="../Classification/drop/index.php">drop</a></td> <td>Drops the model from the Vertica DB.</td> </tr>
    <tr> <td><a href="../Classification/features_importance/index.php">features_importance</a></td> <td>Computes the model features importance by normalizing the LinearSVC hyperplan coefficients.</td> </tr>
    <tr> <td><a href="../Classification/fit/index.php">fit</a></td> <td>Trains the model.</td> </tr>
    <tr> <td><a href="../Classification/lift_chart/index.php">lift_chart</a></td> <td>Draws the model Lift Chart.</td> </tr>
    <tr> <td><a href="../Classification/plot">plot</a></td> <td>Draws the LinearSVC if the number of predictors is lesser than 3.</td> </tr>
    <tr> <td><a href="../Classification/prc_curve/index.php">prc_curve</a></td> <td>Draws the model PRC curve.</td> </tr>
    <tr> <td><a href="../Classification/predict/index.php">predict</a></td> <td>Predicts using the input relation.</td> </tr>
    <tr> <td><a href="../Classification/roc_curve/index.php">roc_curve</a></td> <td>Draws the model ROC curve.</td> </tr>
    <tr> <td><a href="../Classification/score/index.php">score</a></td> <td>Computes the model score.</td> </tr>

</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.svm</span> <span class="k">import</span> <span class="n">LinearSVC</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">LinearSVC</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;iris_svm&quot;</span><span class="p">,</span>
                  <span class="n">tol</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span> 
                  <span class="n">C</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span> 
                  <span class="n">fit_intercept</span> <span class="o">=</span> <span class="kc">True</span><span class="p">,</span>
                  <span class="n">intercept_scaling</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span>
                  <span class="n">intercept_mode</span> <span class="o">=</span> <span class="s2">&quot;regularized&quot;</span><span class="p">,</span>
                  <span class="n">class_weight</span> <span class="o">=</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>
                  <span class="n">max_iter</span> <span class="o">=</span> <span class="mi">100</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;LinearSVC&gt;
</pre>
</div>
</div>

</div>
</div>

</div>