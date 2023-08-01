from django.urls import path
from imagedetect.views import CountObjectsView

urlpatterns = [
    path('detection', CountObjectsView.as_view(), name='detection'),


]