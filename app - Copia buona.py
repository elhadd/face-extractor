import time
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock

class MyApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Pulsante per avviare il processo di elaborazione
        self.start_button = Button(text="Avvia Elaborazione", size_hint=(1, None), height=50,
                                   background_color=(0.13, 0.59, 0.95, 1), color=(1, 1, 1, 1))
        self.start_button.bind(on_press=self.start_processing)
        self.layout.add_widget(self.start_button)

        # Barra di progresso
        self.progress_bar = ProgressBar(value=0, size_hint=(1, None), height=30)
        self.layout.add_widget(self.progress_bar)

        return self.layout

    def start_processing(self, instance):
        # Disabilita il pulsante durante l'elaborazione
        self.start_button.disabled = True

        # Avvia il processo di elaborazione in un thread separato
        threading.Thread(target=self.simulate_processing, daemon=True).start()

    def simulate_processing(self):
        # Simulazione di un processo di elaborazione
        for i in range(1, 11):
            # Simula il tempo di elaborazione
            time.sleep(1)

            # Aggiorna la barra di avanzamento sulla thread principale usando Clock
            Clock.schedule_once(lambda dt: self.update_progress(i * 10), 0)

        # Riabilita il pulsante una volta completata l'elaborazione
        Clock.schedule_once(lambda dt: setattr(self.start_button, 'disabled', False), 0)

    def update_progress(self, value):
        self.progress_bar.value = value

if __name__ == '__main__':
    MyApp().run()
