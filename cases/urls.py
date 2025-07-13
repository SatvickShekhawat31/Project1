from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('inference/', views.inference, name='inference'),  # Route for processing user input
]
