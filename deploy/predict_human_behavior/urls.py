from django.conf.urls   import url
from predict_human_behavior import views
from django.urls import path

urlpatterns = [
    path('model/', views.get)
]