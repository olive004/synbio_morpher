package ur_rna.Utilities.swing;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.JTextComponent;
import java.util.Arrays;
import java.util.function.Consumer;

/**
 * Simplifies management of JTextComponent text-change event handling.
 * Usually to be notified of text-change events, one must register a DocumentListener, which has 3 methods.
 * To complicate matters, a Document may fire events on threads other than the UI thread.
 *
 * This class simplifies the situation by combining all three methods into a single one that fires an event
 * that is guaranteed to be on the UI thread.
 *
 * It also has a boolean {@link #hasChanged()} that returns true if a change event occurred any time after
 * a call to {@link #reset()}.  These calls can be combined in a synchronous way using {@link #resetIfChanged()}.
 *
 * Clients can synchronize on this to prevent prevent its state from changing in-between calls to hasChanged() and reset():
 * {@code
 *  private final TextDocListener docListener = new TextDocListener();
 *  private function doActionIfDocChanged() {
 *      boolean changed;
 *      synchronized(docListener) {
 *          changed = docListener.hasChanged();
 *          if (changed)
 *              docListener.reset();
 *     }
 *     if (changed) { ... }
 *  }
 * }
 *
 * Alternatively, a client can use the {@link #resetIfChanged()} function to both get the state and reset it in a synchronized way:
 * {@code
 *  private function doActionIfDocChanged() {
 *      if (docListener.resetIfChanged()) { ... }
 *  }
 * }
 */
public class TextDocListener implements DocumentListener {
    //Consumer<DocumentEvent> docListener;
    //ActionListener listener;
    Object[] listeners = new Object[1];
    int count = 0;
    boolean changed;
    int suspendCount;

    public TextDocListener(){}
    public TextDocListener(Runnable listener){ addListener(listener); }
    public TextDocListener(Consumer<DocumentEvent> listener){ addListener(listener); }

    /**
     * Add this as a DocumentListener to a JTextComponent's Document.
     */
    public void listen(final JTextComponent txt) {
        txt.getDocument().addDocumentListener(this);
    }

    /**
     * Add a method that will receive DocumentEvents generated by the Documents this class is listening to.
     * @param listener a DocumentEvent Consumer that is called when DocumentEvents are generated.
     */
    public void addListener(Consumer<DocumentEvent> listener) {
        add(listener);
    }

    /**
     * Add a method that will be called whenever the the Documents this class is listening to generate DocumentEvents.
     * @param listener a Runnable that is called when DocumentEvents are generated.
     */
    public void addListener(Runnable listener) {
        add(listener);
    }
    private void add(Object o) {
        if (count >= listeners.length)
            listeners = Arrays.copyOf(listeners, Math.max(4, listeners.length * 2));
        listeners[count++] = o;
    }
    private boolean removeListener(Object listener) {
        if (listener != null)
            for (int i = 0; i < listeners.length; i++) {
                if (listener.equals(listeners[i])) {
                    listeners[i] = null; // free the object (in case the next line doesn't due to no more listeners)
                    System.arraycopy(listeners, i + 1, listeners, i, listeners.length - (i + 1));
                    count--;
                    return true;
                }
            }
        return false;
    }

    @Override
    public void insertUpdate(final DocumentEvent e) {
        update(e);
    }
    @Override
    public void removeUpdate(final DocumentEvent e) {
        update(e);
    }
    @Override
    public void changedUpdate(final DocumentEvent e) {
        update(e);
    }

    public boolean hasChanged() { return changed; }

    public void markChanged() {
        synchronized (this) { // clients can synchronize on this object too, to prevent 'changed' from changing in-between calls to hasChanged() and reset()
            changed = true;
        }
    }
    private void update(final DocumentEvent e) {
        if (suspendCount != 0) return;
        markChanged();
        if (count == 0) return;
        if (SwingUtilities.isEventDispatchThread())
            notifyListeners(e);
        else
            SwingUtilities.invokeLater(()->notifyListeners(e));
    }
    /**
     * Sets the {@link #hasChanged()} state to false and returns the previous value before the reset.
     * The change is performed atomically (synchronized) and allows a client to both get the changed state
     * and reset it without the possiblity of state being changed by another thread in between calls to
     * {@link #hasChanged()} and {@link #reset()}.
     */
    public boolean resetIfChanged() {
        boolean oldVal;
        synchronized (this) {
            oldVal = changed;
            changed = false;
        }
        return oldVal;
    }
    public void reset() {
        changed = false;
    }

    public void suspendUpdates() { suspendCount++; }
    public void resumeUpdates() {
        if (suspendCount==0) throw new IllegalStateException(this.getClass().getSimpleName() +  "function resumeUpdates() was called too many times.");
        suspendCount--;
    }


    @SuppressWarnings("unchecked") // listeners only contains objects of type Runnable or Consumer<DocumentEvent>, so we can ignore the UncheckedCast warning of converting Consumer to Consumer<DocumentEvent>
    private void notifyListeners(final DocumentEvent e) {
        Object[] copy = Arrays.copyOf(listeners, count);
        for (Object o : copy) {
            if (o instanceof Runnable)
                ((Runnable) o).run();
            else
                ((Consumer<DocumentEvent>)o).accept(e);
        }
    }
}