import site
site.addsitedir(r'D:\\mytools\\AI4Water')

from ai4water.experiments import MLRegressionExperiments
from ai4water.utils.utils import dateandtime_now

from utils import get_x_y


antibiotic_name = "cip"  # change to cfx and azm
data = get_x_y(antibiotic_name)


class MyExp(MLRegressionExperiments):

    def build_and_run(self,
                      predict=True,
                      view=False,
                      title=None,
                      fit_kws=None,
                      cross_validate=False,
                      **kwargs):

        """Builds and run one 'model' of the experiment.

        Since and experiment consists of many models, this method
        is also run many times. """

        verbosity = 1

        model = self.ai4water_model(
            data=self.data,
            prefix=title,
            verbosity=verbosity,
            **self.model_kws,
            **kwargs
        )

        setattr(self, '_model', model)

        val_score = model.cross_val_score(model.config['val_metric'])

        tt, tp = model.predict('test')

        #model.view_model()
        #model.explain()
        model.interpret()

        if predict:
            t, p = model.predict('training')

            return (t,p), (tt, tp)

        if model.config['val_metric'] in ['r2', 'nse', 'kge', 'r2_mod']:
            val_score = 1.0 - val_score

        return val_score


myexp = MyExp(
    data= data,
    train_data="random",
    cross_validator = {'KFold': {'n_splits': 10}},
    exp_name = f"{antibiotic_name}_{dateandtime_now()}",
)

myexp.fit(
    run_type='optimize',
    num_iterations=20,
    opt_method="bayes",
    exclude=[
        'model_TheilsenRegressor',   # todo too much time
        'model_RadiusNeighborsRegressor', # todo for SHap Inter
        'model_ElasticNetCV',   # todo, too much time
        'model_ExtraTreesRegressor',  # too much time
        'model_LarsCV',  # error
        'model_OrthogonalMatchingPursuitCV',  # too much time
        'model_RANSACRegressor',   # error

             ],
    cross_validate=True,
)


myexp.plot_cv_scores(include=[
    'model_LGBMRegressor',
    'model_RandomForestRegressor',
    'model_CATBoostRegressor',
    'model_HistGradientBoostingRegressor',
    'model_GradientBoostingRegressor',
    'model_BaggingRegressor',
    'XGBoostRegressor'
],
show=True
)
myexp.compare_errors('r2')
myexp.compare_errors('rmse')
myexp.compare_errors('nse')

myexp.taylor_plot()
myexp.taylor_plot(plot_bias=True, name='taylor_with_bias')




