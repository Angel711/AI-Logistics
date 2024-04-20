from django.urls import path
from . import views
from .views import product_list

urlpatterns = [
    path('products/', product_list, name='product_list'),
]

urlpatterns = [
    path('prueba/', views.Prueba, name='Prueba')
]